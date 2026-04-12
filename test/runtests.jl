using Test

@testset "DataDrivenAcoustics.jl" begin
    include("case1.jl")
    include("case2.jl")
    include("case3.jl")
    include("case4.jl")
end
