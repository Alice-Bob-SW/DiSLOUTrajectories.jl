using Test
using TestReports
using TestReports.EzXML: readxml, root, findall

@testset "JUnit runner preserves test outcomes (package=$package)" for package in (false, true)
    runner = abspath(joinpath(@__DIR__, "..", "junit.jl"))
    project = dirname(Base.active_project())
    mktempdir() do directory
        if package
            environment = TestReports.get_testreports_environment()
            cp(joinpath(environment, "Manifest.toml"), joinpath(directory, "Manifest.toml"))
            write(
                joinpath(directory, "Project.toml"), """
                name = "ReportingFixture"
                uuid = "5bcb7a70-255f-4d06-8e07-26a4d071487a"
                """ * read(joinpath(environment, "Project.toml"), String)
            )
            mkpath(joinpath(directory, "src"))
            write(joinpath(directory, "src", "ReportingFixture.jl"), "module ReportingFixture end")
            mkpath(joinpath(directory, "test"))
        end
        fixture = package ? joinpath(directory, "test", "runtests.jl") : joinpath(directory, "fixture.jl")
        output = joinpath(directory, "junit.xml")
        for (assertion, failures, errors) in
            (("@test true", 0, 0), ("@test false", 1, 0), ("error(\"boom\")", 0, 1))
            write(
                fixture, """
                using Test
                @testset "report <&>" begin
                    # Module-valued metadata, like JET results, cannot be deep-copied.
                    TestReports.record_testset_property!(Test.get_testset(), "module", Main)
                    @test_skip false
                    $assertion
                end
                @testset "afterwards" begin
                    @test true
                end
                """
            )
            test_project = package ? directory : project
            argument = package ? "--package" : fixture
            command = `$(Base.julia_cmd()) --startup-file=no --project=$test_project $runner $argument`
            log = IOBuffer()
            process = run(pipeline(ignorestatus(addenv(command, "JUNIT_OUTPUT" => output)); stdout = log, stderr = log))
            if success(process) != (failures + errors == 0) || !isfile(output)
                print(String(take!(log)))
            end
            @test success(process) == (failures + errors == 0)
            @test isfile(output)
            isfile(output) || continue
            xml = root(readxml(output))
            @test length(findall("//failure", xml)) == failures
            @test length(findall("//error", xml)) == errors
            @test length(findall("//skipped", xml)) == 1
            @test length(findall("//testcase", xml)) == 3
            @test any(node -> occursin("report <&>", node["name"]), findall("//testsuite", xml))
            rm(output)
        end
    end
end
