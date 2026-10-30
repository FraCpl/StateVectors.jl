"""
    StateVectors

Minimal named-component vector and matrix for use with DifferentialEquations.jl.

    x = StateVector{Float64}(pos = 2, vel = 2, E = 1)   # zeros; sizes only
    x.pos .= [1.0, 0.0]       # size n > 1: a view into the data
    x.E = 3.0                 # size 1: a scalar (use `=`, not `.=`)

    # wrap an existing flat vector (no copy)
    x = StateVector([1.0, 0.0, 0.0, 1.0, 0.0]; pos = 2, vel = 2, E = 1)

    # matrices: a row layout and a column layout (columns default to rows)
    J = StateMatrix{Float64}(x)                          # layout taken from x
    P = StateMatrix{Float64}((pos = 2, vel = 2))         # or give sizes
    J[:pos, :vel] .= [1.0 0.0; 0.0 1.0]                  # block views
    J[:E, :E] = 0.5                                      # scalar x scalar

    # Rectangular matrices:
    x = StateVector{Float64}(pos = 2, vel = 2, E = 1)    # 5 states
    z = StateVector{Float64}(y = 2)                      # 2 measurements

    H = StateMatrix{Float64}(z, x)                       # 2×5: rows from z, cols from x
    H[:meas, :pos] .= [1.0 0.0; 0.0 1.0]

    # or
    Q = StateMatrix{Float64}((y = 2,), (pos = 2, vel = 2, E = 1))

"""
module StateVectors

export StateVector, StateMatrix, getdata, getaxes

# ---------------------------------------------------------------------------
# Layouts
#
# A layout ("axis") is a NamedTuple name => Int (size-1 component) or
# UnitRange (size-n component).  It is stored as a type parameter, so lookups
# like `x.pos` are resolved at compile time.
# ---------------------------------------------------------------------------

_layout(::Tuple{}, n) = ()
function _layout(sizes::Tuple, n)
    l = first(sizes)
    r = l == 1 ? n + 1 : (n + 1):(n + l)
    return (r, _layout(Base.tail(sizes), n + l)...)
end

# @generated => the layout is computed at compile time and is a true constant,
# so everything built from it is inferable.
@generated function _axes(::Val{sizes}) where {sizes}
    ax = NamedTuple{keys(sizes)}(_layout(values(sizes), 0))
    return QuoteNode(ax)
end

_axlen(ax::NamedTuple) = sum(length, values(ax))

@inline function _resolve(ax::NamedTuple, s::Symbol)
    haskey(ax, s) || throw(KeyError(s))
    return ax[s]
end
@inline _resolve(::NamedTuple, i) = i

_get(d, i::Int) = d[i]
_get(d, r::AbstractUnitRange) = view(d, r)
_assign!(d, i::Int, v) = (d[i] = v)
_assign!(d, r::AbstractUnitRange, v) = (d[r] .= v)

getdata(x::AbstractArray) = x

# ---------------------------------------------------------------------------
# StateVector
# ---------------------------------------------------------------------------

struct StateVector{T,A<:AbstractVector{T},Ax} <: AbstractVector{T}
    data::A
end

function _wrap(data::A, v::Val{sizes}) where {A<:AbstractVector,sizes}
    length(data) == sum(values(sizes)) ||
        throw(DimensionMismatch("layout needs $(sum(values(sizes))) elements, got $(length(data))"))
    return StateVector{eltype(A),A,_axes(v)}(data)
end

@inline StateVector{T}(; sizes...) where {T} =
    _wrap(zeros(T, sum(values(NamedTuple(sizes)))), Val(NamedTuple(sizes)))

@inline StateVector(data::AbstractVector; sizes...) = _wrap(data, Val(NamedTuple(sizes)))

getdata(x::StateVector) = getfield(x, :data)
getaxes(::StateVector{T,A,Ax}) where {T,A,Ax} = Ax

# x.name
@inline _index(x::StateVector, s::Symbol) = _resolve(getaxes(x), s)

@inline Base.getproperty(x::StateVector, s::Symbol) = _get(getdata(x), _index(x, s))
@inline function Base.setproperty!(x::StateVector, s::Symbol, v)
    _assign!(getdata(x), _index(x, s), v)
    return v
end
Base.propertynames(x::StateVector, ::Bool = false) = keys(getaxes(x))

# AbstractVector interface
Base.size(x::StateVector) = size(getdata(x))
Base.IndexStyle(::Type{<:StateVector}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(x::StateVector, i::Int) = getdata(x)[i]
Base.@propagate_inbounds function Base.setindex!(x::StateVector, v, i::Int)
    getdata(x)[i] = v
    return x
end

Base.getindex(x::StateVector, s::Symbol) = getproperty(x, s)
Base.setindex!(x::StateVector, v, s::Symbol) = (setproperty!(x, s, v); x)
Base.view(x::StateVector, s::Symbol) = view(getdata(x), _index(x, s))

# similar / copy keep the layout (DiffEq calls these constantly)
function Base.similar(x::StateVector{T,A,Ax}, ::Type{S}) where {T,A,Ax,S}
    d = similar(getdata(x), S)
    return StateVector{S,typeof(d),Ax}(d)
end
function Base.copy(x::StateVector{T,A,Ax}) where {T,A,Ax}
    d = copy(getdata(x))
    return StateVector{T,typeof(d),Ax}(d)
end

function Base.show(io::IO, x::StateVector)
    print(io, "StateVector(")
    first = true
    for k in propertynames(x)
        first || print(io, ", ")
        first = false
        v = getproperty(x, k)
        print(io, k, " = ", v isa AbstractArray ? collect(v) : v)
    end
    print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", x::StateVector) = show(io, x)

# ---------------------------------------------------------------------------
# StateMatrix  (row layout Ax1, column layout Ax2)
# ---------------------------------------------------------------------------

struct StateMatrix{T,A<:AbstractMatrix{T},Ax1,Ax2} <: AbstractMatrix{T}
    data::A
end

# layout "tokens": from sizes, or from an existing StateVector
@inline _axval(sizes::NamedTuple) = Val(_axes(Val(sizes)))
@inline _axval(x::StateVector) = Val(getaxes(x))

function _wrapax(data::A, ::Val{ax1}, ::Val{ax2}) where {A<:AbstractMatrix,ax1,ax2}
    size(data) == (_axlen(ax1), _axlen(ax2)) ||
        throw(DimensionMismatch("layout needs $((_axlen(ax1), _axlen(ax2))), got $(size(data))"))
    return StateMatrix{eltype(A),A,ax1,ax2}(data)
end

# zeros, from sizes or from StateVectors (columns default to rows)
@inline function StateMatrix{T}(rows::NamedTuple, cols::NamedTuple = rows) where {T}
    data = zeros(T, sum(values(rows)), sum(values(cols)))
    return _wrapax(data, _axval(rows), _axval(cols))
end
@inline function StateMatrix{T}(rows::StateVector, cols::StateVector = rows) where {T}
    return _wrapax(zeros(T, length(rows), length(cols)), _axval(rows), _axval(cols))
end

# wrap existing data (no copy)
@inline StateMatrix(data::AbstractMatrix, rows::NamedTuple, cols::NamedTuple = rows) =
    _wrapax(data, _axval(rows), _axval(cols))
@inline StateMatrix(data::AbstractMatrix, rows::StateVector, cols::StateVector = rows) =
    _wrapax(data, _axval(rows), _axval(cols))

getdata(M::StateMatrix) = getfield(M, :data)
getaxes(::StateMatrix{T,A,Ax1,Ax2}) where {T,A,Ax1,Ax2} = (Ax1, Ax2)

# AbstractMatrix interface
Base.size(M::StateMatrix) = size(getdata(M))
Base.@propagate_inbounds Base.getindex(M::StateMatrix, i::Int, j::Int) = getdata(M)[i, j]
Base.@propagate_inbounds function Base.setindex!(M::StateMatrix, v, i::Int, j::Int)
    getdata(M)[i, j] = v
    return M
end

# Named blocks: M[:a, :b], M[:a, :], M[:, :b], M[:a, 1:2], ...
# `view` is defined too, because `M[:a, :b] .= x` lowers to a `view` call.
const _Sel = Union{Colon,Integer,AbstractVector{<:Integer}}

@inline function _idx(M::StateMatrix, i, j)
    ax1, ax2 = getaxes(M)
    return _resolve(ax1, i), _resolve(ax2, j)
end

_blk(d, i::Int, j::Int) = d[i, j]                 # scalar x scalar -> number
_blk(d, i, j) = view(d, i, j)
_setblk!(d, v, i::Int, j::Int) = (d[i, j] = v)
_setblk!(d, v, i, j) = (view(d, i, j) .= v)

for (TI, TJ) in ((Symbol, Symbol), (Symbol, _Sel), (_Sel, Symbol))
    @eval begin
        Base.getindex(M::StateMatrix, i::$TI, j::$TJ) = _blk(getdata(M), _idx(M, i, j)...)
        Base.view(M::StateMatrix, i::$TI, j::$TJ) = view(getdata(M), _idx(M, i, j)...)
        function Base.setindex!(M::StateMatrix, v, i::$TI, j::$TJ)
            _setblk!(getdata(M), v, _idx(M, i, j)...)
            return v
        end
    end
end

function Base.similar(M::StateMatrix{T,A,Ax1,Ax2}, ::Type{S}) where {T,A,Ax1,Ax2,S}
    d = similar(getdata(M), S)
    return StateMatrix{S,typeof(d),Ax1,Ax2}(d)
end
function Base.copy(M::StateMatrix{T,A,Ax1,Ax2}) where {T,A,Ax1,Ax2}
    d = copy(getdata(M))
    return StateMatrix{T,typeof(d),Ax1,Ax2}(d)
end

# M * v  -> StateVector carrying M's row layout
function Base.:*(M::StateMatrix{T,A,Ax1,Ax2}, x::AbstractVector) where {T,A,Ax1,Ax2}
    y = getdata(M) * getdata(x)
    return StateVector{eltype(y),typeof(y),Ax1}(y)
end

Base.summary(io::IO, M::StateMatrix) =
    print(io, size(M, 1), "×", size(M, 2), " StateMatrix(rows = ", keys(getaxes(M)[1]),
          ", cols = ", keys(getaxes(M)[2]), ")")

# ---------------------------------------------------------------------------
# Broadcasting: keep the StateVector / StateMatrix type through `@.`, `.=`, ...
# ---------------------------------------------------------------------------

Base.BroadcastStyle(::Type{<:StateVector}) = Broadcast.ArrayStyle{StateVector}()
Base.BroadcastStyle(::Type{<:StateMatrix}) = Broadcast.ArrayStyle{StateMatrix}()
# vector with matrix -> matrix style
Base.BroadcastStyle(::Broadcast.ArrayStyle{StateVector},
                    s::Broadcast.ArrayStyle{StateMatrix}) = s
Base.BroadcastStyle(s::Broadcast.ArrayStyle{StateMatrix},
                    ::Broadcast.ArrayStyle{StateVector}) = s

# find the first argument of type C in a (nested) Broadcasted
_find(::Type{C}, bc::Broadcast.Broadcasted) where {C} = _find(C, bc.args)
_find(::Type{C}, ::Tuple{}) where {C} = nothing
function _find(::Type{C}, args::Tuple) where {C}
    a = _find1(C, first(args))
    return a === nothing ? _find(C, Base.tail(args)) : a
end
_find1(::Type{C}, bc::Broadcast.Broadcasted) where {C} = _find(C, bc.args)
_find1(::Type{C}, x) where {C} = x isa C ? x : nothing

# The result's dimensionality is part of the type, so each method below has a
# statically known return type.

# 1-D -> StateVector
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{StateVector},<:Tuple{Any}},
                      ::Type{T}) where {T}
    x = _find(StateVector, bc)
    axes(bc) == axes(x) ||
        throw(DimensionMismatch("cannot broadcast a StateVector to a different length"))
    return similar(x, T)
end

# 2-D -> StateMatrix
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{StateMatrix},<:Tuple{Any,Any}},
                      ::Type{T}) where {T}
    M = _find(StateMatrix, bc)
    axes(bc) == axes(M) ||
        throw(DimensionMismatch("cannot broadcast a StateMatrix to a different size"))
    return similar(M, T)
end

# anything else (e.g. `x .* y'`, 3-D): no meaningful layout -> plain Array
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{StateVector},<:Tuple{Any,Any,Vararg}},
             ::Type{T}) where {T} = similar(Array{T}, axes(bc))
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{StateMatrix},<:Tuple{Any,Any,Any,Vararg}},
             ::Type{T}) where {T} = similar(Array{T}, axes(bc))

end # module
