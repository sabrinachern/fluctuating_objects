module FluctuatingDiffusion

using LinearAlgebra

mutable struct ObjectSpec
    kind::Symbol
    profile::Symbol
    pos_idx::Int
    extent::Int
    magnitude::Float64
    is_fluctuating::Bool
    Pr::Float64
    is_asymmetric::Bool
    fun_expr::Union{Nothing,String}
    fun_compiled::Union{Nothing,Function}
    force_mode::Symbol
    half_side::Symbol
end

ObjectSpec(kind::Symbol; profile::Symbol=:localized, pos_idx::Int,
           extent::Int=1, magnitude::Real=1.0,
           is_fluctuating::Bool=false, Pr::Real=NaN, is_asymmetric::Bool=false,
           fun_expr::Union{Nothing,String}=nothing, force_mode::Symbol=:flip,
           half_side::Symbol=:right) =
    ObjectSpec(kind, profile, pos_idx, extent, float(magnitude),
               is_fluctuating, float(Pr), is_asymmetric, fun_expr, nothing,
               force_mode, half_side)

mutable struct MobileObjectSpec
    shape::Symbol
    x0::Float64
    radius::Float64
    magnitude::Float64
    mobility::Float64
    harmonic_trap_K::Float64
    temp_ratio::Float64
    t_start_move::Float64
end

MobileObjectSpec(; shape::Symbol=:linear_bump, x0::Real, radius::Real,
                 magnitude::Real=1.0, mobility::Real=0.0, harmonic_trap_K::Real=0.0, temp_ratio::Real=1.0, t_start_move::Real=0.0) =
    MobileObjectSpec(shape, float(x0), float(radius), float(magnitude),
                     float(mobility), float(harmonic_trap_K), float(temp_ratio), float(t_start_move))

mutable struct Param
    dim::Int8
    dt::Float64
    dx::Float64
    dx_field::Float64
    L::Float64
    N::Int64
    K::Float64
    a_cut::Float64
    Pr_default::Float64
    objects::Vector{ObjectSpec}
    mobile_objects::Vector{MobileObjectSpec}
    num_x::Int
    num_field::Int
    periodic::Bool
    measure_rho2::Bool
end

function setParam(dim::Integer, dt::Real, dx::Real, dx_field::Real,
                  L::Real, N::Integer, K::Real, a_cut::Real,
                  Pr_default::Real, objects::Vector{ObjectSpec},
                  mobile_objects::Vector{MobileObjectSpec};
                  periodic::Bool=false, measure_rho2::Bool=true)
    num_x     = max(1, Int(round(L / dx)))
    num_field = max(1, Int(round(L / dx_field)))
    return Param(Int8(dim), float(dt), float(dx), float(dx_field), float(L), Int64(N),
                 float(K), float(a_cut), float(Pr_default), objects, mobile_objects,
                 num_x, num_field, periodic, measure_rho2)
end

mutable struct State
    t::Float64
    pos::Matrix{Float64}
    obj_pos::Vector{Float64}
    fields::Vector{Vector{Float64}}
    ρ::Vector{Float64}
    ρ_sum::Vector{Float64}
    ρ2_sum::Matrix{Float64}
    block_count::Int
    ρ_avg_block::Vector{Float64}
    ρ_sec_avg_block::Matrix{Float64}

    J_interval_sum::Float64
    J_interval_temp::Float64
    J_interval_avg_block::Float64
    J2_interval_avg_block::Float64

    J_cross_sum::Float64
    J_cross_temp::Float64
    J_cross_avg_block::Float64
    J2_cross_avg_block::Float64

    cell_head::Vector{Int}
    next_in_cell::Vector{Int}
    cell_of::Vector{Int}
    cell_size::Float64
    ncell::Int
    nspan::Int
    fint::Vector{Float64}
    obj_fint::Vector{Float64}
end

@inline _obj_Pr(p::Param, o::ObjectSpec) = isnan(o.Pr) ? p.Pr_default : o.Pr
@inline function _wrap(dx::Real, L::Real)
    Δ  = float(dx)
    LL = float(L)
    return Δ - LL * round(Int, Δ / LL)
end

@inline _field_idx(i::Int, nf::Int, periodic::Bool) = periodic ? mod1(i, nf) : clamp(i, 1, nf)
@inline function _x_to_idx(x::Real, dxF::Real, nf::Int)
    return clamp(Int(floor(float(x) / float(dxF))) + 1, 1, nf)
end

@inline function _clamp_x(x::Real, L::Real, periodic::Bool)
    return periodic ? mod(float(x), float(L)) : clamp(float(x), 0.0, float(L))
end

@inline function _localized_max_left_idx(obj::ObjectSpec, nf::Int)
    obj.extent >= 0 || error("Localized object extent must be >= 0; got $(obj.extent).")
    obj.extent <= nf - 1 || error("Localized object extent=$(obj.extent) is too large for num_field=$(nf).")
    return max(1, nf - obj.extent)
end

@inline function _clamp_obj_idx(idx::Int, obj::ObjectSpec, nf::Int, periodic::Bool)
    if periodic
        return mod1(idx, nf)
    elseif obj.profile === :localized
        return clamp(idx, 1, _localized_max_left_idx(obj, nf))
    else
        return clamp(idx, 1, nf)
    end
end

function _maybe_compile_fun!(obj::ObjectSpec)
    obj.profile == :function || return
    obj.fun_expr === nothing && error("profile=:function requires `fun = \"expr in x,a,c,L\"`")
    ex = Meta.parse(obj.fun_expr)
    obj.fun_compiled = eval(:((x,a,c,L) -> ($(ex))))
end

@inline function _in_interval(x::Float64, x_left::Float64, x_right::Float64)
    return (x >= x_left) && (x <= x_right)
end

@inline function _unwrap_dx(xnew::Float64, xold::Float64, L::Float64, periodic::Bool)
    dx = xnew - xold
    return periodic ? _wrap(dx, L) : dx
end


function _init_object_field!(field::Vector{Float64}, obj::ObjectSpec, p::Param, rng)
    nf  = p.num_field
    dxF = p.dx_field
    fill!(field, 0.0)

    if obj.kind == :potential
        if obj.profile == :half_quad
            obj.is_fluctuating && (@warn "half_quad is static; disabling fluctuation."; obj.is_fluctuating = false)
            a = obj.magnitude
            c = obj.pos_idx * dxF
            side = obj.half_side === :left ? :left : :right
            for s in 1:nf
                x = s * dxF
                if side === :right
                    x > c && (field[s] = a * (x - c)^2)
                else
                    x < c && (field[s] = a * (c - x)^2)
                end
            end
        elseif obj.profile == :localized
            if obj.is_fluctuating
                if obj.is_asymmetric
                    config = rand(rng, [1,2]) #rand(rng, [1,2,3,4]) # this will tell you how many configurations it switches between
                    for k in 0:obj.extent-1 
                        i = mod1(obj.pos_idx+k, nf)
                        if config == 1 # left bump
                            field[i] = obj.magnitude - obj.magnitude/(obj.extent-1) * k
                        elseif config == 2 # right bump
                            if k == obj.extent-1
                                field[i] = 0
                            else
                                field[i] = obj.magnitude/(obj.extent-1) * (k+1)
                            end
                        elseif config == 3 # left well
                            field[i] = - obj.magnitude + obj.magnitude/(obj.extent-1) * k
                        elseif config == 4 # right well
                            if k == obj.extent-1
                                field[i] = 0
                            else
                                field[i] = -obj.magnitude/(obj.extent-1) * (k+1)
                            end
                        else
                            error("Unknown potential profile")
                        end
                    #field[i] = 2 * obj.magnitude * (rand(rng) - 0.5)
                    end
                else # if object is symmetric
                    amp = obj.magnitude * rand(rng, [1, -1])
                    profile = [0.5, 1, 0.5]
                    for k in 1:obj.extent-1
                        i = _field_idx(obj.pos_idx + k, nf, p.periodic)
                        field[i] = amp * profile[k]
                    end
                end
            else
                if obj.extent <= 1
                    field[_field_idx(obj.pos_idx, nf, p.periodic)] = obj.magnitude
                else
                    center = obj.extent / 2
                    for k in 0:obj.extent
                        i = _field_idx(obj.pos_idx + k, nf, p.periodic)
                        frac = 1 - abs(k - center) / max(center, 1)
                        field[i] = obj.magnitude * max(frac, 0.0)
                    end
                end
            end
        elseif obj.profile == :function
            _maybe_compile_fun!(obj)
            a = obj.is_fluctuating ? 2 * obj.magnitude * (rand(rng) - 0.5) : obj.magnitude
            c = (obj.pos_idx - 1) * dxF
            f = obj.fun_compiled::Function
            for s in 1:nf
                x = (s - 1) * dxF
                field[s] = f(x, a, c, p.L)
            end
        else
            error("Unknown potential profile: $(obj.profile)")
        end
    elseif obj.kind == :force
        obj.profile == :localized || @warn "Forces only support :localized; using :localized."
        signed_val = if obj.is_fluctuating
            if obj.force_mode === :flip
                rand(rng) < 0.5 ? -obj.magnitude : obj.magnitude
            elseif obj.force_mode === :amplitude
                2 * obj.magnitude * (rand(rng) - 0.5)
            else
                error("Unknown force_mode $(obj.force_mode); use :flip or :amplitude")
            end
        else
            obj.magnitude
        end
        for k in 0:obj.extent
            i = _field_idx(obj.pos_idx + k, nf, p.periodic)
            field[i] = signed_val
        end
    else
        error("Unknown object kind $(obj.kind).")
    end
    return nothing
end

function _pair_cutoff(p::Param)
    mobile_cut = isempty(p.mobile_objects) ? 0.0 : maximum(obj.radius for obj in p.mobile_objects)
    return max(abs(p.K) > 0 ? p.a_cut : 0.0, mobile_cut)
end

function setState(p::Param, rng; t0::Real=0.0)
    pos = zeros(Float64, p.N, p.dim)
    for n in 1:p.N, d in 1:p.dim
        pos[n, d] = rand(rng) * p.L
    end

    obj_pos = [_clamp_x(obj.x0, p.L, p.periodic) for obj in p.mobile_objects]

    fields = [zeros(Float64, p.num_field) for _ in p.objects]
    for (j, obj) in enumerate(p.objects)
        if !(obj.kind == :potential && obj.profile == :localized)
            obj.is_asymmetric = false
        end
        obj.pos_idx = _clamp_obj_idx(obj.pos_idx, obj, p.num_field, p.periodic)
        _init_object_field!(fields[j], obj, p, rng)
    end

    ρ            = zeros(Float64, p.num_x)
    ρ_sum        = zeros(Float64, p.num_x)
    ρ2_sum       = zeros(Float64, p.num_x, p.num_x)
    ρ_avg_block  = zeros(Float64, p.num_x)
    ρ_sec_avg_bl = zeros(Float64, p.num_x, p.num_x)

    J_interval_sum = 0.0
    J_interval_temp = 0.0
    J_interval_avg_block = 0.0
    J2_interval_avg_block = 0.0

    J_cross_sum = 0.0
    J_cross_temp = 0.0
    J_cross_avg_block = 0.0
    J2_cross_avg_block = 0.0

    total_entities = p.N + length(p.mobile_objects)
    pair_cut = _pair_cutoff(p)
    if pair_cut > 0
        cs     = max(pair_cut, 1e-12)
        ncell  = max(1, Int(ceil(p.L / cs)))
        nspan  = max(1, Int(ceil(pair_cut / cs)))
        heads  = fill(0, ncell)
        nextin = fill(0, total_entities)
        cellof = fill(0, total_entities)
        fint   = zeros(Float64, p.N)
        obj_fint = zeros(Float64, length(p.mobile_objects))
    else
        cs = 0.0; ncell = 0; nspan = 0
        heads = Int[]; nextin = Int[]; cellof = Int[]
        fint = zeros(Float64, p.N)
        obj_fint = zeros(Float64, length(p.mobile_objects))
    end

    return State(float(t0), pos, obj_pos, fields, ρ,
             ρ_sum, ρ2_sum, 0,
             ρ_avg_block, ρ_sec_avg_bl,
             J_interval_sum, J_interval_temp, J_interval_avg_block, J2_interval_avg_block,
             J_cross_sum, J_cross_temp, J_cross_avg_block, J2_cross_avg_block,
             heads, nextin, cellof, cs, ncell, nspan, fint, obj_fint)
end

@inline function _cell_id(x::Float64, L::Float64, inv_cs::Float64, ncell::Int, periodic::Bool)
    xc = periodic ? (x - L * floor(x / L)) : (x < 0 ? 0.0 : (x >= L ? prevfloat(L) : x))
    ui = Int(floor(xc * inv_cs))
    ui = if ui < 0
        0
    elseif ui >= ncell
        ncell - 1
    else
        ui
    end
    return 1 + ui
end

@inline _entity_is_particle(p::Param, eid::Int) = eid <= p.N
@inline _mobile_index(p::Param, eid::Int) = eid - p.N
@inline function _entity_x(p::Param, s::State, eid::Int)
    return _entity_is_particle(p, eid) ? s.pos[eid, 1] : s.obj_pos[_mobile_index(p, eid)]
end

function _build_cells!(p::Param, s::State)
    s.ncell == 0 && return nothing
    inv_cs = 1 / s.cell_size
    fill!(s.cell_head, 0)
    fill!(s.next_in_cell, 0)

    for n in 1:p.N
        c = _cell_id(s.pos[n, 1], p.L, inv_cs, s.ncell, p.periodic)
        s.cell_of[n] = c
        s.next_in_cell[n] = s.cell_head[c]
        s.cell_head[c] = n
    end

    for k in 1:length(p.mobile_objects)
        eid = p.N + k
        c = _cell_id(s.obj_pos[k], p.L, inv_cs, s.ncell, p.periodic)
        s.cell_of[eid] = c
        s.next_in_cell[eid] = s.cell_head[c]
        s.cell_head[c] = eid
    end
    return nothing
end

@inline function _add_particle_force!(s::State, n::Int, f::Float64)
    s.fint[n] += f
end

@inline function _add_mobile_force!(s::State, k::Int, f::Float64)
    s.obj_fint[k] += f
end

@inline function _linear_bump_force(rel::Float64, radius::Float64, magnitude::Float64)
    if radius <= 0 || abs(rel) >= radius || rel == 0.0
        return 0.0
    end
    return magnitude / radius * sign(rel)
end

function _interactions!(p::Param, s::State)
    fill!(s.fint, 0.0)
    fill!(s.obj_fint, 0.0)
    s.ncell == 0 && return nothing

    _build_cells!(p, s)
    pair_pp_on = abs(p.K) > 0
    a2 = p.a_cut^2

    for c in 1:s.ncell
        n = s.cell_head[c]
        while n != 0
            xn = _entity_x(p, s, n)
            for off in -s.nspan:s.nspan
                cc = p.periodic ? mod1(c + off, s.ncell) : (c + off)
                if !p.periodic && (cc < 1 || cc > s.ncell)
                    continue
                end
                m = s.cell_head[cc]
                while m != 0
                    if m > n
                        xm = _entity_x(p, s, m)
                        dx = p.periodic ? _wrap(xm - xn, p.L) : (xm - xn)

                        n_is_particle = _entity_is_particle(p, n)
                        m_is_particle = _entity_is_particle(p, m)

                        if n_is_particle && m_is_particle
                            if pair_pp_on && dx * dx < a2
                                f = 2 * p.K * dx
                                _add_particle_force!(s, n, -f)
                                _add_particle_force!(s, m,  f)
                            end
                        elseif n_is_particle != m_is_particle
                            if n_is_particle
                                part_idx = n
                                mob_idx  = _mobile_index(p, m)
                                rel = _entity_x(p, s, n) - _entity_x(p, s, m)
                            else
                                part_idx = m
                                mob_idx  = _mobile_index(p, n)
                                rel = _entity_x(p, s, m) - _entity_x(p, s, n)
                            end
                            rel = p.periodic ? _wrap(rel, p.L) : rel
                            mob = p.mobile_objects[mob_idx]
                            f_part = mob.shape === :linear_bump ? _linear_bump_force(rel, mob.radius, mob.magnitude) : 0.0
                            _add_particle_force!(s, part_idx, f_part)
                            _add_mobile_force!(s, mob_idx, -f_part)
                        end
                    end
                    m = s.next_in_cell[m]
                end
            end
            n = s.next_in_cell[n]
        end
    end
    return nothing
end

@inline function _field_drift_at_x(p::Param, s::State, x::Float64)
    nf = p.num_field
    i  = _x_to_idx(x, p.dx_field, nf)
    i2 = p.periodic ? (i == nf ? 1 : i + 1) : min(i + 1, nf)
    inv_dxF = 1.0 / p.dx_field
    drift = 0.0
    for (j, obj) in enumerate(p.objects)
        if obj.kind == :potential
            drift += (s.fields[j][i] - s.fields[j][i2]) * inv_dxF
        else
            drift += s.fields[j][i]
        end
    end
    return drift
end

@inline function _mobile_obj_field_drift_at_x(p::Param, s::State, x::Float64, r::Float64, mag::Float64)
    nf = p.num_field
    points = [x - r, x - (r/2) , x, x + (r/2), x + r]
    drift = 0.0
    for point in points    
        if point < x  
            mag_for_point = (mag / r) * (point - (x-r))
        else
            mag_for_point = (mag / r) * ((x+r) - point)
        end
        i  = _x_to_idx(point, p.dx_field, nf)
        i2 = p.periodic ? (i == nf ? 1 : i + 1) : min(i + 1, nf)
        inv_dxF = 1.0 / p.dx_field
        for (j, obj) in enumerate(p.objects)
            if obj.kind == :potential
                drift += mag_for_point * (s.fields[j][i] - s.fields[j][i2]) * inv_dxF
            else
                drift += mag_for_point * s.fields[j][i]
            end
        end
    end
    return drift
end

# function update!(p::Param, s::State, t_run::Real, rng)
function update!(p::Param, s::State, t_run::Real, rng;
             measure_interval_current::Bool=false,
             x_left::Float64=0.0,
             x_right::Float64=0.0, 
             x_measure::Float64=0.0)
    steps = max(1, Int(round(t_run / p.dt)))
    σ = sqrt(2 * p.dt)

    @inbounds for _ in 1:steps
        _interactions!(p, s)

        s.J_interval_temp = 0.0
        s.J_cross_temp = 0.0
        for n in 1:p.N
            drift = _field_drift_at_x(p, s, s.pos[n, 1]) + s.fint[n]
            xold = s.pos[n, 1]
            xnew = xold + p.dt * drift + σ * randn(rng)

            if measure_interval_current
                accumulate_interval_current!(p, s, xold, xnew, x_left, x_right)
                accumulate_cross_current!(p, s, xold, xnew, x_measure)
            end
            s.pos[n, 1] = _clamp_x(xnew, p.L, p.periodic)
        end

        if measure_interval_current
            # after measuring the integration of J(x,t) at each time t, add this to the sum and also the squared sum
            s.J_interval_sum += s.J_interval_temp / steps
            # s.J2_interval_sum += s.J_interval_temp^2 / steps
    
            s.J_cross_sum += s.J_cross_temp / steps
            # s.J2_cross_sum += s.J_cross_temp^2 / steps
        end

        for (j, obj) in enumerate(p.objects)
            if obj.is_fluctuating && rand(rng) < _obj_Pr(p, obj)
                _init_object_field!(s.fields[j], obj, p, rng)
            end
        end

        t_next = s.t + p.dt
        for k in eachindex(p.mobile_objects)
            mob = p.mobile_objects[k]
            if mob.mobility == 0.0 || t_next < mob.t_start_move
                continue
            end 
            drift = s.obj_fint[k] + _mobile_obj_field_drift_at_x(p, s, s.obj_pos[k], mob.radius, mob.magnitude)
            drift += - mob.harmonic_trap_K * (s.obj_pos[k] - mob.x0) ## -dU/dx for harmonic trap potential
            xnew = s.obj_pos[k] + p.dt * mob.mobility * drift + sqrt(2 * p.dt * mob.mobility * mob.temp_ratio) * randn(rng)
            s.obj_pos[k] = _clamp_x(xnew, p.L, p.periodic)
        end

        s.t += p.dt
    end
    return nothing
end

function accumulate_cross_current!(
    p::Param,
    s::State,
    xold::Float64,
    xnew::Float64,
    x_measure::Float64
)
    dx = _unwrap_dx(xnew, xold, p.L, p.periodic)
    
    if xold < xnew
        if xold < x_measure && x_measure < xnew 
            s.J_cross_temp += dx
        end
    else
        if xnew < x_measure && x_measure < xold 
            s.J_cross_temp += dx
        end
    end
    return nothing
end

function accumulate_interval_current!(
    p::Param,
    s::State,
    xold::Float64,
    xnew::Float64,
    x_left::Float64,
    x_right::Float64
)
    dx = _unwrap_dx(xnew, xold, p.L, p.periodic)
    xmid = xold + 0.5 * dx

    if p.periodic
        xmid = mod(xmid, p.L)
    end

    if _in_interval(xmid, x_left, x_right)
        s.J_interval_temp += dx
    end
    return nothing
end


function _fill_density!(p::Param, s::State)
    fill!(s.ρ, 0.0)
    invdx = 1.0 / p.dx
    @inbounds for n in 1:p.N
        idx = 1 + Int(floor(s.pos[n, 1] * invdx))
        idx = max(1, min(idx, p.num_x))
        s.ρ[idx] += 1.0
    end
end

function getρ_and_ρsq!(p::Param, s::State)
    _fill_density!(p, s)
    s.block_count += 1
    @inbounds @simd for i in 1:p.num_x
        s.ρ_sum[i] += s.ρ[i]
    end
    if p.measure_rho2
        LinearAlgebra.BLAS.syr!('U', 1.0, s.ρ, s.ρ2_sum)
    end
    return nothing
end

function finalize_block!(p::Param, s::State)
    B = s.block_count
    B == 0 && return 0
    invB = 1.0 / B

    @inbounds @simd for i in 1:p.num_x
        s.ρ_avg_block[i] = s.ρ_sum[i] * invB
    end

    fill!(s.ρ_sec_avg_block, 0.0)
    if p.measure_rho2
        @inbounds for j in 1:p.num_x
            @simd for i in j:p.num_x
                s.ρ_sec_avg_block[j, i] = s.ρ2_sum[j, i] * invB
            end
        end
    end
    
    # average local current over interval
    s.J_interval_avg_block = s.J_interval_sum * invB 
    s.J2_interval_avg_block = s.J_interval_sum.^2 * invB

    s.J_cross_avg_block = s.J_cross_sum * invB
    s.J2_cross_avg_block = s.J_cross_sum.^2 * invB

    s.J_interval_sum = 0.0
    s.J_cross_sum = 0.0
    
    fill!(s.ρ_sum, 0.0)
    fill!(s.ρ2_sum, 0.0)
    s.block_count = 0
    return B
end

function get_forces!(p::Param, s::State, f::AbstractVector{Float64})
    fill!(f, 0.0)
    inv_dxF = 1.0 / p.dx_field
    nf = p.num_field
    @inbounds for n in 1:p.N
        i  = _x_to_idx(s.pos[n, 1], p.dx_field, nf)
        i2 = p.periodic ? (i == nf ? 1 : i + 1) : min(i + 1, nf)
        for (j, obj) in enumerate(p.objects)
            if obj.kind == :potential
                f[j] += -(s.fields[j][i] - s.fields[j][i2]) * inv_dxF
            else
                f[j] += -s.fields[j][i]
            end
        end
    end
    return f
end

function get_forces(p::Param, s::State)
    f = zeros(Float64, length(p.objects))
    get_forces!(p, s, f)
    return f, f .^ 2
end

function init_positions_from_array!(p::Param, s::State, pos::AbstractMatrix{<:Real}, rng)
    size(pos, 1) == p.N  || error("pos snapshot has N=$(size(pos,1)) but current run needs N=$(p.N)")
    size(pos, 2) == p.dim || error("pos snapshot has dim=$(size(pos,2)) but current run needs dim=$(p.dim)")

    @inbounds for n in 1:p.N, d in 1:p.dim
        s.pos[n, d] = _clamp_x(Float64(pos[n, d]), p.L, p.periodic)
    end
    _build_cells!(p, s)
    return nothing
end

end # module
