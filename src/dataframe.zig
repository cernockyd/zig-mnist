const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const win = std.os.windows;

pub const Data = struct {
    labels_df: DataFrame,
    values_df: DataFrame,

    pub fn deinit(self: Data) void {
        self.labels_df.deinit();
        self.values_df.deinit();
    }
};

const DEBUG = false;

pub const DataFrameError = error{
    NullData,
    IndexOutOfBounds,
    ShapeMismatch,
    NotSingleColumn,
};

pub const Shape = struct {
    m: usize,
    n: usize,
};

pub const DataFrame = struct {
    shape: Shape,
    data: ?[]f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, m: usize, n: usize, data: ?[]f32) !DataFrame {
        const d = if (data) |dat| (dat) else try allocator.alloc(f32, m * n);
        errdefer allocator.free(d);
        return DataFrame{ .shape = Shape{ .m = m, .n = n }, .allocator = allocator, .data = d };
    }

    pub fn info(self: DataFrame) void {
        if (self.data == null) {
            std.debug.print("DataFrame is empty\n", .{});
            return;
        }
        std.debug.print("Matrix: m = {}, n = {}\n", .{ self.shape.m, self.shape.n });
    }

    pub fn deinit(self: DataFrame) void {
        if (self.data != null) {
            self.allocator.free(self.data.?);
        }
    }

    pub fn rand(allocator: Allocator, m: usize, n: usize) !DataFrame {
        var prng = std.rand.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.posix.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });
        const rnd = prng.random();
        const data = try allocator.alloc(f32, m * n);
        errdefer allocator.free(data);
        var i: usize = 0;
        while (i < m * n) : (i += 1) {
            // https://ziglang.org/documentation/master/std/#std.Random.floatNorm
            data[i] = rnd.floatNorm(f32) * 0.2;
        }
        return DataFrame{ .shape = Shape{ .m = m, .n = n }, .allocator = allocator, .data = data };
    }

    // fischer-yates shuffle
    pub fn shuffle_rows(self: DataFrame) !void {
        if (self.data) |data| {
            var prng = std.rand.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                try std.posix.getrandom(std.mem.asBytes(&seed));
                break :blk seed;
            });
            const rnd = prng.random();
            var j: usize = undefined;
            const tmp: []f32 = try self.allocator.alloc(f32, self.shape.n);
            defer self.allocator.free(tmp);
            var i: usize = 0;
            while (i < self.shape.m) : (i += 1) {
                j = rnd.intRangeAtMost(usize, 0, i);
                const j_start = (j * self.shape.n);
                const j_end = (j + 1) * self.shape.n;
                const i_start = (i * self.shape.n);
                const i_end = (i + 1) * self.shape.n;
                if (i != j) {
                    @memcpy(tmp, data[j_start..j_end]);
                    @memcpy(data[j_start..j_end], data[i_start..i_end]);
                    @memcpy(data[i_start..i_end], tmp);
                }
            }
            return;
        }
        return DataFrameError.NullData;
    }

    pub fn copy(self: DataFrame) !DataFrame {
        const data = try self.allocator.alloc(f32, self.shape.m * self.shape.n);
        errdefer self.allocator.free(data);
        @memcpy(data, self.data.?);
        return DataFrame{ .shape = self.shape, .allocator = self.allocator, .data = data };
    }

    pub fn get(self: DataFrame, y: u32, x: u32) f32 {
        return self.data.?[self.shape.n * y + x];
    }

    pub fn set(self: DataFrame, y: u32, x: u32, val: f32) void {
        self.data.?[self.shape.n * y + x] = val;
    }

    pub fn apply(self: DataFrame, comptime a: fn (val: f32, idx: u32) f32) !void {
        if (self.data == null) {
            return DataFrameError.NullData;
        }
        var i: u32 = 0;
        while (i < self.shape.m * self.shape.n) : (i += 1) {
            self.data.?[i] = a(self.data.?[i], i);
        }
    }

    // naive implementation
    pub fn dot(self: DataFrame, b: DataFrame) !DataFrame {
        if (self.shape.n != b.shape.m) {
            std.debug.print("Matrix A columns count must be equal to Matrix B rows\n", .{});
            return DataFrameError.ShapeMismatch;
        }
        if (self.data == null or b.data == null) {
            return DataFrameError.NullData;
        }
        const shape = Shape{ .m = self.shape.m, .n = b.shape.n };
        // should use try DataFrame.init();
        const data = try self.allocator.alloc(f32, shape.m * shape.n);
        errdefer self.allocator.free(data);
        var i: u32 = 0;
        const m = self.shape.m;
        const n = self.shape.n;
        const p = b.shape.n;
        while (i < m) : (i += 1) {
            var j: u32 = 0;
            while (j < p) : (j += 1) {
                var s: f32 = 0.0; // sum
                var k: u32 = 0;
                while (k < n) : (k += 1) {
                    // print(
                    //     "i = {d}, j = {d}, k = {d}\n\n",
                    //     .{i, j, k},
                    // );
                    // print("{d} * {d}\n", .{self.get(i, k).?, b.get(k, j).?});
                    s += self.get(i, k) * b.get(k, j);
                }
                data[p * i + j] = s;
            }
        }
        return DataFrame{
            .shape = shape,
            .allocator = self.allocator,
            .data = data,
        };
    }

    // naive implementation
    pub fn doti(self: DataFrame, a: DataFrame, b: DataFrame) !void {
        if (a.shape.n != b.shape.m) {
            std.debug.print("Matrix A columns count must be equal to Matrix B rows\n", .{});
            return DataFrameError.ShapeMismatch;
        }
        if (self.shape.n != b.shape.m) {
            std.debug.print("Result matrix columns count must be equal to Matrix B rows\n", .{});
            return DataFrameError.ShapeMismatch;
        }
        if (self.data == null or a.data == null or b.data == null) {
            return DataFrameError.NullData;
        }
        var i: u32 = 0;
        const m = self.shape.m;
        const n = self.shape.n;
        const p = b.shape.n;
        while (i < m) : (i += 1) {
            var j: u32 = 0;
            while (j < p) : (j += 1) {
                var s: f32 = 0.0; // sum
                var k: u32 = 0;
                while (k < n) : (k += 1) {
                    // print(
                    //     "i = {d}, j = {d}, k = {d}\n\n",
                    //     .{i, j, k},
                    // );
                    // print("{d} * {d}\n", .{self.get(i, k).?, b.get(k, j).?});
                    s += self.get(i, k).? * b.get(k, j).?;
                }
                self.data[p * i + j] = s;
            }
        }
    }

    // pub fn sub(self: DataFrame, b: DataFrame) !DataFrame {
    //     if (self.shape.n != b.shape.n or self.shape.m != b.shape.m) {
    //         std.debug.print("Matrix A shape must be equal to Matrix B's\n", .{});
    //         return DataFrameError.ShapeMismatch;
    //     }
    //     if (self.data == null or b.data == null) {
    //         return DataFrameError.NullData;
    //     }
    //     const result = try DataFrame.init(self.allocator, self.shape.m, self.shape.n, null);
    //     var i: u32 = 0;
    //     while (i < self.shape.m) : (i += 1) {
    //         result.data.?[i] = self.data.?[i] - b.data.?[i];
    //     }
    //     return result;
    // }

    pub fn sum(self: DataFrame, comptime T: type) !T {
        var s: T = 0.0;
        if (self.data) |data| {
            for (data) |val| {
                s += val;
            }
            return s;
        }
        return DataFrameError.NullData;
    }

    pub fn pow(self: DataFrame, y: f32) !void {
        if (self.data == null) {
            return DataFrameError.NullData;
        }
        var i: u32 = 0;
        while (i < self.shape.m * self.shape.n) : (i += 1) {
            self.data.?[i] = std.math.pow(f32, self.data.?[i], y);
        }
    }

    // This function could be faster if read byte by byte and parse the number
    pub fn load_csv(path: []const u8, allocator: Allocator) !DataFrame {
        std.log.info("Loading CSV file: {s}", .{path});

        const file: std.fs.File = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const len = try file.getEndPos();

        const contents = try file.reader().readAllAlloc(
            allocator,
            len,
        );
        defer allocator.free(contents);

        const data = try allocator.alloc(f32, len);
        var m: u32 = 0;
        var i: u32 = 0;
        var n: u32 = 0;
        var lines = std.mem.split(u8, contents, "\n");

        // get number of columns
        const first_line = lines.peek();
        if (first_line) |line| {
            var num = std.mem.split(u8, line, ",");
            while (num.next()) |_| {
                n += 1;
            }
        }

        // get number of rows and fill data
        while (lines.next()) |line| {
            if (line.len == 0) {
                break;
            }
            m += 1;
            var values = std.mem.split(u8, line, ",");
            while (values.next()) |value| {
                const parsed = std.fmt.parseFloat(f32, value) catch 0; // default to 0
                data[i] = parsed;
                i += 1;
            }
        }

        print("info: loaded {} values\n", .{m});
        print("info: Example {d:6.5}, {d:6.5} values\n", .{ data[0], data[1] });

        return DataFrame{
            .shape = Shape{ .m = m, .n = n },
            .allocator = allocator,
            .data = data,
        };
    }

    pub fn save_txt(self: DataFrame, path: []const u8) !void {
        const file: std.fs.File = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        for (self.data.?, 0..) |val, i| {
            if (i % self.shape.n == 0 and i != 0) {
                try file.write("\n");
            }
            try file.write("{f},", .{val});
        }
    }

    pub fn clip(self: DataFrame) !void {
        if (self.data == null) {
            return DataFrameError.NullData;
        }
        for (self.data.?, 0..) |val, i| {
            if (val > 1.0) {
                self.data.?[i] = 1.0;
            } else if (val < (-1.0)) {
                self.data.?[i] = -1.0;
            }
        }
    }

    pub fn slice(
        self: DataFrame,
        end: usize,
        start: ?usize,
    ) error{ OutOfMemory, IndexOutOfBounds }!DataFrame {
        var column_start_index: usize = 0;
        if (start) |s| {
            column_start_index = s;
        }
        const new_n: usize = end - column_start_index;
        print("info: Slicing from {d} to {d}, total columns = {d}\n", .{ column_start_index, end, new_n });

        const new_data = try self.allocator.alloc(f32, self.shape.m * new_n);
        errdefer self.allocator.free(new_data);

        if (column_start_index >= end or end >= self.shape.m) {
            return DataFrameError.IndexOutOfBounds;
        }

        var row_index: usize = 0;
        if (self.data) |data| {
            while (row_index < self.shape.m) : (row_index += 1) { // iterate all rows
                var column_new_index: usize = 0;
                var column_original_index: usize = column_start_index;
                while (column_new_index < end and column_original_index < end) : (column_new_index += 1) { // end slicing
                    // print("row: {d}, new_n: {d}, column_new_index: {d}\n", .{row_index, new_n, column_new_index});
                    new_data[row_index * new_n + column_new_index] = data[row_index * self.shape.n + column_original_index];
                    column_original_index += 1;
                }
            }
        }

        return DataFrame{ .shape = Shape{ .m = self.shape.m, .n = new_n }, .allocator = self.allocator, .data = new_data };
    }

    pub fn head(self: DataFrame, n: usize) void {
        var max_rows = n;
        const max_columns = 20;
        if (self.data == null) {
            print("DataFrame is empty.", .{});
            return;
        }
        if (self.data) |data| {
            if (n > self.shape.m) {
                max_rows = self.shape.m;
            }
            var i: u32 = 0;
            while (i < max_rows) : (i += 1) {
                var j: u32 = 0;
                print("# {d:3}: ", .{i});
                while (j < self.shape.n and j <= max_columns) : (j += 1) {
                    print("{d:3} ", .{data[i * self.shape.n + j]});
                }
                print("\n", .{});
            }
        }
    }

    // element-wise sigmoid
    pub fn sigmoid(self: DataFrame) void {
        if (self.data) |data| {
            for (data, 0..) |val, index| {
                data[index] = (1.0 / (1.0 + std.math.exp(-val)));
            }
        }
    }

    pub fn sigmoid_derivative(self: DataFrame) void {
        if (self.data) |data| {
            for (data, 0..) |val, index| {
                const sigm = (1.0 / (1.0 + std.math.exp(-val)));
                data[index] = sigm * (1.0 - sigm);
            }
        }
    }

    // vector-wise softmax (row-wise)
    pub fn softmax(self: DataFrame) void {
        var i: u32 = 0;
        if (self.data) |data| {
            while (i < self.shape.m) : (i += 1) {
                var t: f32 = 0; // total
                // slice to get the row
                const s = i * self.shape.n;
                const e = s + self.shape.n;
                const row = data[s..e];
                for (row, 0..) |val, index| {
                    t += std.math.exp(val);
                    _ = index;
                }
                for (row, 0..) |val, index| {
                    row[index] = std.math.exp(val) / t;
                }
            }
        }
    }

    pub fn max_index(self: DataFrame) usize {
        var max_idx: usize = 0;
        if (self.data) |data| {
            for (self.data, 0..) |val, index| {
                if (val > data[max_idx]) {
                    max_idx = index;
                }
            }
        }
        return max_idx;
    }

    pub fn normalize(self: DataFrame, min: f32, max: f32) void {
        if (self.data) |data| {
            for (data, 0..) |val, index| {
                data[index] = (val - min) / (max - min) * 0.2;
            }
        }
    }

    pub fn sub(self: DataFrame, substractor: DataFrame) !void {
        if (self.data == null or substractor.data == null) {
            return DataFrameError.NullData;
        }
        if (self.shape.m != substractor.shape.m) {
            return DataFrameError.ShapeMismatch;
        }
        if (self.shape.n != substractor.shape.n) {
            return DataFrameError.ShapeMismatch;
        }
        for (self.data.?, 0..) |val, index| {
            self.data.?[index] = val - substractor.data.?[index];
        }
    }

    pub fn add(self: DataFrame, addend: DataFrame) !void {
        if (self.data == null or addend.data == null) {
            return DataFrameError.NullData;
        }
        if (self.shape.m != addend.shape.m) {
            return DataFrameError.ShapeMismatch;
        }
        if (self.shape.n != addend.shape.n) {
            return DataFrameError.ShapeMismatch;
        }
        for (self.data.?, 0..) |val, index| {
            self.data.?[index] = val + addend.data.?[index];
        }
    }

    pub fn transpose(self: DataFrame) !DataFrame {
        if (self.data == null) {
            return DataFrameError.NullData;
        }
        const new_df = try DataFrame.init(self.allocator, self.shape.n, self.shape.m, null);
        errdefer new_df.deinit();
        if (self.data) |data| {
            for (data, 0..) |val, index| {
                const row = index / self.shape.n;
                const col = index % self.shape.n;
                new_df.data.?[col * self.shape.m + row] = val;
            }
        }
        return new_df;
    }

    // element-wise multiplication
    pub fn mul(self: DataFrame, multiplier: DataFrame) !void {
        if (self.data == null or multiplier.data == null) {
            return DataFrameError.NullData;
        }
        if (self.shape.m != multiplier.shape.m) {
            return DataFrameError.ShapeMismatch;
        }
        if (self.shape.n != multiplier.shape.n) {
            return DataFrameError.ShapeMismatch;
        }
        for (self.data.?, 0..) |val, index| {
            self.data.?[index] = val * multiplier.data.?[index];
        }
    }

    pub fn one_hot(self: DataFrame, len: u32) !DataFrame {
        if (self.data == null) {
            return DataFrameError.NullData;
        }
        if (self.shape.n != 1) {
            std.debug.print("Matrix must have a single column\n", .{});
            return DataFrameError.NotSingleColumn;
        }
        const encoded = try DataFrame.init(self.allocator, self.shape.m, len, null);
        for (self.data.?, 0..) |label, row_i| {
            var col: u32 = 0;
            while (col < len) : (col += 1) {
                const float_col: f32 = @as(f32, @floatFromInt(col));
                const value: f32 = if (label == float_col) 1.0 else 0.0;
                encoded.set(@as(u32, @intCast(row_i)), col, value);
            }
        }
        return encoded;
    }
};

pub const BatchIterator = struct {
    index: usize = 0,
    batch_size: usize,
    data: Data = undefined,

    pub fn next(self: *BatchIterator) ?Data {
        if (self.index * self.batch_size >= self.data.values_df.shape.m - 1 or self.data.values_df.data == null or self.index * self.batch_size >= self.data.labels_df.shape.m - 1 or self.data.labels_df.data == null) {
            return null;
        }
        self.index += 1;
        const labels_slice = self.data.labels_df.data.?[(self.index * self.data.labels_df.shape.n)..((self.index + self.batch_size) * self.data.labels_df.shape.n)];
        return Data{
            .labels_df = DataFrame{
                .allocator = self.data.labels_df.allocator,
                .shape = Shape{ .m = self.batch_size, .n = self.data.labels_df.shape.n },
                .data = labels_slice,
            },
            .values_df = DataFrame{
                .allocator = self.data.values_df.allocator,
                .shape = Shape{ .m = self.batch_size, .n = self.data.values_df.shape.n },
                .data = self.data.values_df.data.?[(self.index * self.data.values_df.shape.n)..((self.index + self.batch_size) * self.data.values_df.shape.n)],
            },
        };
    }
};
