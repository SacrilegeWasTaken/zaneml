const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("zaneml", .{
        .root_source_file = b.path("lib/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const example_imports: []const std.Build.Module.Import = &.{
        .{ .name = "zaneml", .module = mod },
    };

    const main_exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = example_imports,
        }),
    });

    const perceptron_exe = b.addExecutable(.{
        .name = "perceptron",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/perceptron.zig"),
            .target = target,
            .optimize = optimize,
            .imports = example_imports,
        }),
    });

    b.installArtifact(main_exe);

    const run_main_cmd = b.addRunArtifact(main_exe);
    run_main_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_main_cmd.addArgs(args);

    const run_perceptron_cmd = b.addRunArtifact(perceptron_exe);
    if (b.args) |args| run_perceptron_cmd.addArgs(args);

    const run_step = b.step("run", "Run examples/main.zig");
    run_step.dependOn(&run_main_cmd.step);

    const run_main_step = b.step("main", "Run examples/main.zig");
    run_main_step.dependOn(&run_main_cmd.step);

    const run_perceptron_step = b.step("perceptron", "Run examples/perceptron.zig");
    run_perceptron_step.dependOn(&run_perceptron_cmd.step);

    const transformer_exe = b.addExecutable(.{
        .name = "transformer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/transformer.zig"),
            .target = target,
            .optimize = optimize,
            .imports = example_imports,
        }),
    });

    const run_transformer_cmd = b.addRunArtifact(transformer_exe);
    if (b.args) |args| run_transformer_cmd.addArgs(args);

    const run_transformer_step = b.step("transformer", "Run examples/transformer.zig");
    run_transformer_step.dependOn(&run_transformer_cmd.step);

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_mod_tests.step);
}
