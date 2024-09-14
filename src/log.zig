const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;

pub const Log = struct {
    file: std.fs.File,

    pub fn init(name: []const u8) !Log {
        const file = try std.fs.cwd().createFile(
            name,
            .{ .read = true, .truncate = true },
        );
        return Log{ .file = file };
    }

    pub fn write(self: Log, bytes: []const u8) !void {
        try self.file.writeAll(bytes);
    }

    pub fn deinit(self: Log) void {
        self.file.close();
    }
};
