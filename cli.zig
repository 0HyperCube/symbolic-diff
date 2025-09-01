const std = @import("std");
const parser = @import("parser.zig");

pub fn main() !void {
    std.debug.print("Welcome to the CLI\n", .{});
    while (true) {
        std.debug.print(">> ", .{});
        const value = try std.io.getStdIn().reader().readUntilDelimiterAlloc(std.heap.page_allocator, '\n', 1000);
        var tokeniser = parser.Tokeniser.new(value);
        while (try tokeniser.next()) |token| {
            std.debug.print("token {?}\n", .{token});
        }
    }
}
