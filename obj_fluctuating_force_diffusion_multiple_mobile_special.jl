using Random
using Dates
using ProgressMeter
using HDF5
using TOML
using JSON
import Printf: @sprintf

include("obj_module_mobile_special_objects.jl")
using .FluctuatingDiffusion

const d_create = HDF5.create_dataset

_get(d::Dict, k, default) = haskey(d, k) ? d[k] : default
_asvec(x) = isa(x, AbstractVector) ? x : [x]
pretty_t(sec::Real) = string(Dates.canonicalize(Dates.Millisecond(round(Int, sec * 1000))))

function compute_Pr_default(dt; x_ch=nothing, x_ch_over_L=nothing, L=nothing, Pr_default=nothing)
    Pr_default !== nothing && return float(Pr_default)
    if x_ch === nothing && x_ch_over_L !== nothing && L !== nothing
        x_ch = L * x_ch_over_L
    end
    x_ch === nothing && error("Provide either Pr_default or x_ch (or x_ch_over_L + L).")
    return float(dt / (x_ch^2))
end

tojson(::Nothing) = "null"
tojson(x::Bool) = x ? "true" : "false"
tojson(x::Real) = string(x)
tojson(s::AbstractString) = "\"" * replace(replace(s, '\\' => "\\\\"), '"' => "\\\"") * "\""
tojson(v::AbstractVector) = "[" * join((tojson(x) for x in v), ",") * "]"
function tojson(d::Dict{String,Any})
    "{" * join((tojson(k) * ":" * tojson(v) for (k, v) in d), ",") * "}"
end

function parse_args(args::Vector{String})
    cfg_path::Union{Nothing,String} = nothing
    combo::Union{Nothing,Int} = tryparse(Int, get(ENV, "COMBO_INDEX", ""))
    save_every::Int = 1000
    restart_h5::Union{Nothing,String} = nothing
    restart_snapshot::Union{Nothing,String} = nothing

    i = 1
    while i ≤ length(args)
        a = args[i]
        if a == "--config"
            cfg_path = args[i + 1]; i += 2
        elseif startswith(a, "--config=")
            cfg_path = split(a, "=", limit=2)[2]; i += 1
        elseif a == "--combo-index"
            combo = parse(Int, args[i + 1]); i += 2
        elseif startswith(a, "--combo-index=")
            combo = parse(Int, split(a, "=", limit=2)[2]); i += 1
        elseif a == "--save-n-steps"
            save_every = parse(Int, args[i + 1]); i += 2
        elseif startswith(a, "--save-n-steps=")
            save_every = parse(Int, split(a, "=", limit=2)[2]); i += 1
        elseif a == "--restart-h5"
            restart_h5 = args[i + 1]; i += 2
        elseif startswith(a, "--restart-h5=")
            restart_h5 = split(a, "=", limit=2)[2]; i += 1
        elseif a == "--restart-snapshot"
            restart_snapshot = args[i + 1]; i += 2
        elseif startswith(a, "--restart-snapshot=")
            restart_snapshot = split(a, "=", limit=2)[2]; i += 1
        else
            error("Unknown arg: $a")
        end
    end

    cfg_path === nothing && error("Please pass --config path/to/params.toml")
    return cfg_path, something(combo, 1), save_every, restart_h5, restart_snapshot
end

function _expand_ranges!(tbl::Dict{String,Any})
    rng_keys = [k for k in keys(tbl) if endswith(k, "_range")]
    for rk in rng_keys
        base = rk[1:end-6]
        spec = tbl[rk]::Dict
        haskey(spec, "start") || error("$(rk) needs 'start'")
        haskey(spec, "stop") || error("$(rk) needs 'stop'")
        s = Int(spec["start"])
        e = Int(spec["stop"])
        st = Int(get(spec, "step", 1))
        st == 0 && error("$(rk): step cannot be 0")
        tbl[base] = collect(s:st:e)
        delete!(tbl, rk)
    end
end

function _combo_from_cfg(cfg::Dict{String,Any}, combo_index::Int)
    if haskey(cfg, "sweep_list")
        lst = cfg["sweep_list"]::Vector
        total = length(lst)
        1 <= combo_index <= total || error("combo-index out of range 1..$total for sweep_list")
        sel = Dict{String,Any}(pairs(lst[combo_index]))
        return sel, total, "list", collect(keys(sel))
    end

    have_zip = haskey(cfg, "sweep_zip")
    have_cart = haskey(cfg, "sweep")

    zip_keys, zip_arrays, zip_len = String[], Any[], 1
    if have_zip
        z = deepcopy(cfg["sweep_zip"]::Dict)
        _expand_ranges!(z)
        zip_keys = collect(keys(z))
        zip_arrays = [_asvec(z[k]) for k in zip_keys]
        lens = length.(zip_arrays)
        all(lens .== lens[1]) || error("[sweep_zip] arrays must be equal length; got $(lens)")
        zip_len = lens[1]
    end

    cart_keys, cart_arrays, cart_total = String[], Any[], 1
    if have_cart
        s = deepcopy(cfg["sweep"]::Dict)
        _expand_ranges!(s)
        cart_keys = collect(keys(s))
        cart_arrays = [_asvec(s[k]) for k in cart_keys]
        cart_total = isempty(cart_arrays) ? 1 : prod(length.(cart_arrays))
    end

    if !have_zip && !have_cart
        return Dict{String,Any}(), 1, "none", String[]
    end

    total = zip_len * cart_total
    1 <= combo_index <= total || error("combo-index out of range 1..$total")

    idx0 = combo_index - 1
    zip_idx = have_zip ? (idx0 ÷ cart_total) + 1 : 1
    cart_idx = have_cart ? (idx0 % cart_total) + 1 : 1

    sel = Dict{String,Any}()
    if have_zip
        for (i, k) in enumerate(zip_keys)
            sel[k] = zip_arrays[i][zip_idx]
        end
    end
    if have_cart
        strides = [prod(length.(cart_arrays[(i + 1):end])) for i in 1:length(cart_arrays)]
        idxc = cart_idx - 1
        for (i, k) in enumerate(cart_keys)
            stride = i < length(strides) ? strides[i] : 1
            j = (stride == 0 ? 0 : Int(floor(idxc / stride))) % length(cart_arrays[i])
            sel[k] = cart_arrays[i][j + 1]
            idxc -= j * stride
        end
    end

    mode = have_zip && have_cart ? "zip+cartesian" : (have_zip ? "zip" : "cartesian")
    return sel, total, mode, vcat(zip_keys, cart_keys)
end

function pos_to_x(p, L, b_input)
    if p isa AbstractString
        s = strip(p)
        lowercase(s) == "center" && return L / 2
        if (m = match(r"^\s*([A-Za-z])\s*([+\-])\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$", s)) !== nothing
            var = lowercase(m.captures[1])
            op = m.captures[2]
            delta = parse(Float64, m.captures[3])
            base = var == "l" ? L : var == "b" ? b_input : error("unknown var: $var")
            return op == "+" ? base + delta : base - delta
        end
        if (m = match(r"^\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([+\-])\s*([A-Za-z])\s*$", s)) !== nothing
            delta = parse(Float64, m.captures[1])
            op = m.captures[2]
            var = lowercase(m.captures[3])
            base = var == "l" ? L : var == "b" ? b_input : error("unknown var: $var")
            return op == "+" ? delta + base : delta - base
        end
        return parse(Float64, s)
    end
    return Float64(p)
end

function pos_to_idx(p, L, b_input, dxF; num_field=nothing)
    num_field === nothing && (num_field = Int(round(L / dxF)))
    return clamp(Int(round(pos_to_x(p, L, b_input) / dxF)), 1, num_field)
end

# Resolve "$key" placeholders from the selected sweep combo (sel), falling back to cfg.
function resolve_sweep_ref(x, sel::Dict{String,Any}, cfg::Dict{String,Any})
    if x isa AbstractString
        s = strip(x)
        if startswith(s, "\$")
            key = s[2:end]
            haskey(sel, key) && return sel[key]
            haskey(cfg, key) && return cfg[key]
            error("Unknown sweep placeholder \$$key (not found in sel or cfg).")
        end
    end
    return x
end

function main()
    cfg_path, combo_index, save_n_steps, restart_h5_cli, restart_snapshot_cli = parse_args(ARGS)
    cfg = TOML.parsefile(cfg_path)

    sel, total_combos, sweep_mode, swept_keys = _combo_from_cfg(cfg, combo_index)
    println("Combo $combo_index / $total_combos  (mode=$sweep_mode)")
    isempty(sel) || println("Selected parameters: ", sel)

    dim = Int(_get(sel, "dim", _get(cfg, "dim", 1)))
    dt = float(_get(sel, "dt", _get(cfg, "dt", 1e-2)))
    dx = float(_get(sel, "dx", _get(cfg, "dx", 1.0)))
    dxF = float(_get(sel, "dx_field", _get(cfg, "dx_field", dx)))
    L = float(_get(sel, "L", _get(cfg, "L", 100.0)))
    N = Int(_get(sel, "N", _get(cfg, "N", 100)))
    K = float(_get(sel, "K", _get(cfg, "K", 0.0)))
    a_cut = float(_get(sel, "a_cut", _get(cfg, "a_cut", 1.0)))
    periodic = Bool(_get(sel, "periodic", _get(cfg, "periodic", false)))
    measure_rho2 = Bool(_get(sel, "measure_rho2", _get(cfg, "measure_rho2", true)))
    b_input = float(_get(sel, "b", _get(cfg, "b", 0.0)))
    measure_interval_current = Bool(_get(sel, "measure_interval_current",
                                     _get(cfg, "measure_interval_current", false)))
    x_left  = float(_get(sel, "x_left",  _get(cfg, "x_left",  0.0)))
    x_right = float(_get(sel, "x_right", _get(cfg, "x_right", L)))
    x_measure = float(_get(sel, "x_measure", _get(cfg, "x_measure", L/2)))

    x_ch = haskey(cfg, "x_ch") ? float(cfg["x_ch"]) : (haskey(sel, "x_ch") ? float(sel["x_ch"]) : nothing)
    x_ch_over_L = haskey(cfg, "x_ch_over_L") ? float(cfg["x_ch_over_L"]) : (haskey(sel, "x_ch_over_L") ? float(sel["x_ch_over_L"]) : nothing)
    Pr_default_cfg = haskey(cfg, "Pr_default") ? float(cfg["Pr_default"]) : (haskey(sel, "Pr_default") ? float(sel["Pr_default"]) : nothing)
    Pr_default = compute_Pr_default(dt; x_ch=x_ch, x_ch_over_L=x_ch_over_L, L=L, Pr_default=Pr_default_cfg)

    objs_cfg = _get(sel, "objects", _get(cfg, "objects", Any[]))
    mobile_objs_cfg = _get(sel, "mobile_objects", _get(cfg, "mobile_objects", Any[]))
    objs_cfg isa Vector || error("objects must be an array of tables in TOML")
    mobile_objs_cfg isa Vector || error("mobile_objects must be an array of tables in TOML")

    if haskey(cfg, "t_eq") || haskey(sel, "t_eq")
        t_eq = float(_get(sel, "t_eq", _get(cfg, "t_eq", 0.0)))
        t_eq_source, t_eq_factor_used = "absolute", nothing
    else
        t_eq_factor_used = float(_get(sel, "t_eq_factor", _get(cfg, "t_eq_factor", 10.0)))
        t_eq = t_eq_factor_used * L^2
        t_eq_source = "factor"
    end
    if haskey(cfg, "t_between_meas") || haskey(sel, "t_between_meas")
        t_between_meas = float(_get(sel, "t_between_meas", _get(cfg, "t_between_meas", 0.0)))
        t_bm_source, t_bm_factor_used = "absolute", nothing
    else
        t_bm_factor_used = float(_get(sel, "t_between_meas_factor", _get(cfg, "t_between_meas_factor", 1/3)))
        t_between_meas = t_bm_factor_used * L
        t_bm_source = "factor"
    end

    num_field = Int(round(L / dxF))

    objects = FluctuatingDiffusion.ObjectSpec[]
    for o in objs_cfg
        kind = Symbol(o["kind"])
        profile = Symbol(get(o, "profile", "localized"))
        pos_idx = clamp(pos_to_idx(get(o, "pos", "center"), L, b_input, dxF), 1, num_field)
        extent = Int(get(o, "extent", 1))
        magnitude = float(resolve_sweep_ref(get(o, "magnitude", 1.0), sel, cfg))
        fluctuating = Bool(get(o, "fluctuating", false))
        Pr_val = get(o, "Pr", "auto")
        Pr = Pr_val == "auto" ? NaN : float(Pr_val)
        is_asymmetric = Bool(get(o, "is_asymmetric", false))
        fun_expr = haskey(o, "fun") ? String(o["fun"]) : nothing
        force_mode = Symbol(get(o, "force_mode", "flip"))
        half_side = Symbol(get(o, "half_side", "right"))
        push!(objects, FluctuatingDiffusion.ObjectSpec(kind; profile=profile, pos_idx=pos_idx,
               extent=extent, magnitude=magnitude, is_fluctuating=fluctuating,
               Pr=Pr, is_asymmetric=is_asymmetric, fun_expr=fun_expr,
               force_mode=force_mode, half_side=half_side))
    end

    mobile_objects = FluctuatingDiffusion.MobileObjectSpec[]
    for o in mobile_objs_cfg
        shape = Symbol(get(o, "shape", "linear_bump"))
        x0 = pos_to_x(get(o, "pos", "center"), L, b_input)
        radius = float(resolve_sweep_ref(get(o, "radius", 1.0), sel, cfg))
        magnitude = float(resolve_sweep_ref(get(o, "magnitude", 1.0), sel, cfg))
        mobility = float(resolve_sweep_ref(get(o, "mobility", 0.0), sel, cfg))
        harmonic_trap_K = float(resolve_sweep_ref(get(o, "harmonic_trap_K", 0.0), sel, cfg))
        temp_ratio = float(resolve_sweep_ref(get(o, "temp_ratio", 1.0), sel, cfg))
        t_start_move_abs = float(resolve_sweep_ref(get(o, "t_start_move_abs", get(o, "t_start", 0.0)), sel, cfg))
        t_start_move = float(resolve_sweep_ref(get(o, "t_start_move_factor", get(o, "t_start", 0.0)), sel, cfg)) * L^2
        if t_start_move == 0
            t_start_move = t_start_move_abs
        end
        push!(mobile_objects, FluctuatingDiffusion.MobileObjectSpec(; shape=shape, x0=x0,
              radius=radius, magnitude=magnitude, mobility=mobility, harmonic_trap_K = harmonic_trap_K, temp_ratio=temp_ratio,
              t_start_move=t_start_move))
    end

    for o in objects
        if !(o.kind == :potential && o.profile == :localized)
            o.is_asymmetric = false
        end
        if o.kind == :potential && o.profile == :half_quad && o.is_fluctuating
            @warn "half_quad is static; disabling fluctuation at pos_idx=$(o.pos_idx)."
            o.is_fluctuating = false
        end
        if o.kind != :potential && o.profile == :function
            @warn "profile=:function is for potentials; switching to :localized."
            o.profile = :localized
        end
    end

    n_sample = Int(_get(sel, "n_sample", _get(cfg, "n_sample", 10_000)))
    output_dir = _get(sel, "output_dir", _get(cfg, "output_dir", "data"))
    tag = _get(sel, "tag", _get(cfg, "tag", "run"))
    skip_equilibration = Bool(_get(sel, "skip_equilibration", _get(cfg, "skip_equilibration", false)))
    steps_eq = max(1, Int(round(t_eq / dt)))

    seed = Int(_get(sel, "seed", _get(cfg, "seed", 1)))
    rng = MersenneTwister(seed)
    param = FluctuatingDiffusion.setParam(dim, dt, dx, dxF, L, N, K, a_cut, Pr_default,
                                          objects, mobile_objects;
                                          periodic=periodic, measure_rho2=measure_rho2)
    state = FluctuatingDiffusion.setState(param, rng; t0=0.0)

    restart_h5 = restart_h5_cli === nothing ? _get(cfg, "restart_h5", nothing) : restart_h5_cli
    restart_snapshot = restart_snapshot_cli === nothing ? _get(cfg, "restart_snapshot", nothing) : restart_snapshot_cli
    if restart_h5 !== nothing && restart_h5 != ""
        println("Restarting particle positions from: $restart_h5 (snapshot=$(restart_snapshot === nothing ? "eq" : restart_snapshot))")
        h5open(String(restart_h5), "r") do fr
            pos_snap = read(fr["pos_snap"])
            idx = restart_snapshot === nothing || restart_snapshot == "eq" ? 1 : restart_snapshot == "last" ? size(pos_snap, 1) : parse(Int, restart_snapshot)
            idx = clamp(idx, 1, size(pos_snap, 1))
            FluctuatingDiffusion.init_positions_from_array!(param, state, @view(pos_snap[idx, :, :]), rng)
        end
        println("Restart complete at t = ", state.t)
    end

    if skip_equilibration
        println("Skipping equilibration as requested (skip_equilibration=true).")
    else
        println("Equilibrating for steps_eq = $steps_eq (t_eq = $t_eq)...")
        pbar_eq = Progress(steps_eq; desc="equil")
        done = 0
        eq_chunk = max(1, steps_eq ÷ 100)
        first = min(eq_chunk, steps_eq - done)
        t_first = @elapsed FluctuatingDiffusion.update!(param, state, first * dt, rng)
        done += first
        eta_eq = t_first * (steps_eq / first - 1.0)
        println("Estimated equilibration time: ", pretty_t(t_first + eta_eq))
        update!(pbar_eq, done)
        while done < steps_eq
            this = min(eq_chunk, steps_eq - done)
            @elapsed FluctuatingDiffusion.update!(param, state, this * dt, rng)
            done += this
            update!(pbar_eq, done)
        end
        println("Equilibration complete at t = ", state.t)
    end

    FluctuatingDiffusion.finalize_block!(param, state)

    mkpath(output_dir)
    outfile = joinpath(output_dir, @sprintf("L=%d_N=%d_K=%.2f_tag=%s_combo=%d_seed=%d.h5",
        Int(round(L)), N, K, String(tag), combo_index, seed))

    n_blocks = cld(n_sample, save_n_steps)
    num_x = param.num_x
    nobj = length(param.objects)
    nmobile = length(param.mobile_objects)

    objs_dicts = [Dict(
        "kind" => String(o.kind),
        "profile" => String(o.profile),
        "pos_idx" => o.pos_idx,
        "extent" => o.extent,
        "magnitude" => o.magnitude,
        "is_fluctuating" => o.is_fluctuating,
        "Pr" => (isnan(o.Pr) ? "auto" : o.Pr),
        "is_asymmetric" => o.is_asymmetric,
        "force_mode" => String(o.force_mode),
        "half_side" => String(o.half_side),
        "fun_expr" => (o.fun_expr === nothing ? "" : String(o.fun_expr))
    ) for o in param.objects]

    mobile_objs_dicts = [Dict(
        "shape" => String(o.shape),
        "x0" => o.x0,
        "radius" => o.radius,
        "magnitude" => o.magnitude,
        "mobility" => o.mobility,
        "harmonic_trap_K" => o.harmonic_trap_K, 
        "temp_ratio" => o.temp_ratio, 
        "t_start_move" => o.t_start_move
    ) for o in param.mobile_objects]

    params_dict = Dict{String,Any}(
        "config_path" => String(cfg_path),
        "timestamp" => string(now()),
        "dim" => Int(param.dim),
        "dt" => param.dt,
        "dx" => param.dx,
        "dx_field" => param.dx_field,
        "L" => param.L,
        "N" => param.N,
        "K" => param.K,
        "b" => b_input,
        "a_cut" => param.a_cut,
        "periodic" => param.periodic,
        "x_ch" => x_ch === nothing ? nothing : float(x_ch),
        "x_ch_over_L" => x_ch_over_L === nothing ? nothing : float(x_ch_over_L),
        "Pr_default" => param.Pr_default,
        "t_eq" => t_eq,
        "t_between_meas" => t_between_meas,
        "timing_source_eq" => t_eq_source,
        "timing_source_meas" => t_bm_source,
        "t_eq_factor" => t_eq_factor_used,
        "t_between_meas_factor" => t_bm_factor_used,
        "n_sample" => n_sample,
        "save_n_steps" => save_n_steps,
        "n_blocks" => n_blocks,
        "num_x" => num_x,
        "num_field" => param.num_field,
        "seed" => seed,
        "tag" => tag,
        "combo_index" => combo_index,
        "sweep_mode" => sweep_mode,
        "swept_keys" => swept_keys,
        "skip_equilibration" => skip_equilibration,
        "restart_h5" => restart_h5 === nothing ? "" : String(restart_h5),
        "restart_snapshot" => restart_snapshot === nothing ? "" : restart_snapshot,
        "objects" => objs_dicts,
        "mobile_objects" => mobile_objs_dicts,
        "measure_rho2" => measure_rho2, 
        "measure_interval_current" => measure_interval_current, 
        "x_left" => x_left, 
        "x_right" => x_right, 
        "x_measure" => x_measure
    )

    h5open(outfile, "w") do f
        d_create(f, "rho_avg", HDF5.datatype(Float64), HDF5.dataspace((n_blocks, num_x)); chunk=(1, num_x), compress=3)
        d_create(f, "rho_sec_avg", HDF5.datatype(Float64), HDF5.dataspace((n_blocks, num_x, num_x)); chunk=(1, num_x, num_x), compress=1)
        write(f, "params", [tojson(params_dict)])
        d_create(f, "pos_snap", HDF5.datatype(Float64), HDF5.dataspace((n_blocks + 1, N, dim)); chunk=(1, N, dim), compress=1)
        if nobj > 0
            d_create(f, "force", HDF5.datatype(Float64), HDF5.dataspace((n_blocks, nobj)); chunk=(1, nobj), compress=0)
            d_create(f, "force_corr", HDF5.datatype(Float64), HDF5.dataspace((n_blocks, nobj)); chunk=(1, nobj), compress=0)
            d_create(f, "field", HDF5.datatype(Float64), HDF5.dataspace((n_blocks, param.num_field)); chunk=(1, param.num_field), compress=1)
            d_create(f, "field_j", HDF5.datatype(Float64), HDF5.dataspace((n_blocks, nobj, param.num_field)); chunk=(1, nobj, param.num_field), compress=1)
        end
        if nmobile > 0
            d_create(f, "mobile_obj_pos_snap", HDF5.datatype(Float64), HDF5.dataspace((n_blocks + 1, nmobile)); chunk=(1, nmobile), compress=1)
            d_create(f, "mobile_obj_pos_snap_sq", HDF5.datatype(Float64), HDF5.dataspace((n_blocks + 1, nmobile)); chunk=(1, nmobile), compress=1)
        end
        if measure_interval_current
            d_create(f, "J_interval_sum", HDF5.datatype(Float64),
                     HDF5.dataspace((n_blocks,)); chunk=(1,), compress=0)
            d_create(f, "J_interval_sq", HDF5.datatype(Float64),
                     HDF5.dataspace((n_blocks,)); chunk=(1,), compress=0)
            d_create(f, "J_cross_sum", HDF5.datatype(Float64),
                     HDF5.dataspace((n_blocks,)); chunk=(1,), compress=0)
            d_create(f, "J_cross_sq", HDF5.datatype(Float64),
                     HDF5.dataspace((n_blocks,)); chunk=(1,), compress=0)
        end
    end

    println("Measuring in blocks of $save_n_steps samples → $outfile")
    printed_eta = false
    refined_eta_printed = false
    compute_time_accum = 0.0
    total_samples_done = 0

    h5open(outfile, "r+") do f
        f["pos_snap"][1, :, :] = state.pos
        if nmobile > 0
            f["mobile_obj_pos_snap"][1, :] = state.obj_pos
            f["mobile_obj_pos_snap_sq"][1, :] = (state.obj_pos[1])^2
        end
    end

    for b in 1:n_blocks
        block_target = b < n_blocks ? save_n_steps : (n_sample - save_n_steps * (n_blocks - 1))
        println("Block $b/$n_blocks: measuring $block_target samples…")
        pbar_block = Progress(block_target; desc="meas $b/$n_blocks")

        forces_block_sum = zeros(Float64, nobj)
        forces_corr_block_sum = zeros(Float64, nobj)
        for _ in 1:block_target
            t_step = @elapsed begin
                FluctuatingDiffusion.update!(param, state, t_between_meas, rng;
                                            measure_interval_current=measure_interval_current,
                                            x_left=x_left, x_right=x_right, x_measure = x_measure)
                FluctuatingDiffusion.getρ_and_ρsq!(param, state)
                f, f2 = FluctuatingDiffusion.get_forces(param, state)
                forces_block_sum .+= f
                forces_corr_block_sum .+= f2
            end
            compute_time_accum += t_step
            total_samples_done += 1
            if !printed_eta
                eta_meas = t_step * (n_sample - 1)
                println("Estimated measurement time (compute-only): ", pretty_t(t_step + eta_meas))
                printed_eta = true
            end
            next!(pbar_block)
        end

        B = FluctuatingDiffusion.finalize_block!(param, state)
        B == 0 && error("Internal: empty block encountered at block $b")
        force_block_avg = forces_block_sum ./ B
        forces_corr_block_avg = forces_corr_block_sum ./ B

        field_last_total = zeros(Float64, param.num_field)
        field_last_byobj = Array{Float64}(undef, nobj, param.num_field)
        @inbounds for j in 1:nobj
            @views field_last_byobj[j, :] = state.fields[j]
            @views field_last_total .+= state.fields[j]
        end
        
        t_write = @elapsed h5open(outfile, "r+") do f
            f["rho_avg"][b, :] = state.ρ_avg_block
            f["rho_sec_avg"][b, :, :] = state.ρ_sec_avg_block
            if nobj > 0
                f["force"][b, :] = force_block_avg
                f["force_corr"][b, :] = forces_corr_block_avg
                f["field"][b, :] = field_last_total
                f["field_j"][b, :, :] = field_last_byobj
            end
            if measure_interval_current
                f["J_interval_sum"][b] = state.J_interval_avg_block
                f["J_interval_sq"][b] = state.J2_interval_avg_block
                f["J_cross_sum"][b] = state.J_cross_avg_block
                f["J_cross_sq"][b] = state.J2_cross_avg_block
            end
            f["pos_snap"][b + 1, :, :] = state.pos
            if nmobile > 0
                f["mobile_obj_pos_snap"][b + 1, :] = state.obj_pos
                f["mobile_obj_pos_snap_sq"][b + 1, :] = state.obj_pos[1]^2
            end
        end

        if !refined_eta_printed
            per_samp = compute_time_accum / total_samples_done
            blocks_remaining = n_blocks - b
            eta_remaining = per_samp * (n_sample - total_samples_done) + t_write * blocks_remaining
            println("Estimated measurement time (refined incl. writes): ", pretty_t(t_write + eta_remaining))
            refined_eta_printed = true
        end
    end

    println("Done → ", outfile)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
