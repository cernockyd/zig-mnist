const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const DataFrame = @import("dataframe.zig").DataFrame;
const Sequential = @import("sequential.zig").Sequential;
const Data = @import("dataframe.zig").Data;
const visualize = @import("visualize.zig");

pub fn main() !void {
    // try visualize.gradientToImg(null);
    var timer = try std.time.Timer.start();
    const start = timer.lap();
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // const allocator = std.heap.page_allocator;
    // const allocator = std.testing.allocator;
    const data_min: f32 = 0;
    const data_max: f32 = 255;

    // load training data data
    print("\n\nLoad data\n---------\n", .{});
    const mnist_df = try DataFrame.load_csv("./data/mnist_train.csv", allocator);
    print("\nBefore Shuffle\n", .{});
    mnist_df.head(5);
    try mnist_df.shuffle_rows();
    print("\nAfter Shuffle\n", .{});
    mnist_df.head(5);
    print("\nFull\n", .{});
    mnist_df.info();
    mnist_df.head(5);
    print("\nLabels\n", .{});
    var labels_df = try mnist_df.slice(1, 0);
    labels_df.info();
    labels_df.head(5);
    print("\nValues\n", .{});
    var values_df = try mnist_df.slice(mnist_df.shape.n, 1);
    mnist_df.deinit();
    values_df.info();
    values_df.head(5);
    print("\nNormalized\n", .{});
    values_df.normalize(data_min, data_max);
    values_df.head(5);

    const mnist_test_df = try DataFrame.load_csv("./data/mnist_test.csv", allocator);
    defer mnist_test_df.deinit();
    var labels_test_df = try mnist_test_df.slice(1, 0);
    defer labels_test_df.deinit();
    var values_test_df = try mnist_test_df.slice(mnist_test_df.shape.n, 1);
    defer values_test_df.deinit();

    const hidden_1 = 500;
    const hidden_2 = 50;
    const output = 10;
    const theta_1 = try DataFrame.rand(allocator, values_df.shape.n, hidden_1);
    print("\nTheta 1\n", .{});
    theta_1.info();
    theta_1.head(5);
    const theta_2 = try DataFrame.rand(allocator, hidden_1, hidden_2);
    print("\nTheta 2\n", .{});
    theta_2.info();
    const theta_3 = try DataFrame.rand(allocator, hidden_2, output);
    print("\nTheta 3\n", .{});
    theta_3.info();

    // print("\nPrediction batch \n----\n", .{});
    // const predictions = try Sequential.predict(allocator, &values_df, &theta_1, &theta_2, &theta_3);
    // _ = predictions;
    // _ = try Sequential.cost(labels_df, predictions);
    const model = Sequential{ .theta_1 = theta_1, .theta_2 = theta_2, .theta_3 = theta_3 };
    defer model.deinit();
    const train_data = Data{ .labels_df = labels_df, .values_df = values_df };
    defer train_data.deinit();
    const test_data = Data{ .labels_df = labels_test_df, .values_df = values_test_df };
    defer test_data.deinit();
    try model.train(allocator, train_data, test_data);
    //try Sequential.test(allocator, values_df, labels_df, theta_1, theta_2, theta_3);
    // const hidden_input_1 = try values_df.dot(theta_1);
    // hidden_input_1.info();
    // _ = Sequential.init(allocator);
    // {
    //     const hidden_1 = 64;
    //     const theta_1 = try DataFrame.rand(allocator, values_df.shape.n, hidden_1);
    //     const hidden_input_1 = try values_df.dot(theta_1);
    //     hidden_input_1.info();
    //     // const activation = weighted.relu();
    //     // train
    //     // instead I will initialize the
    //     // predict
    // }

    const end = timer.lap();
    const elapsed_micros = @as(f64, @floatFromInt(end - start)) / std.time.ns_per_s;
    // print("dataframe loaded: {d} rows, {d} columns\n", .{ mnist_df.shape.m, mnist_df.shape.n });
    print("execution time:  {d:.3} s\n", .{elapsed_micros});
}

const expect = @import("std").testing.expect;

test "initiate dataframe" {
    _ = try DataFrame.rand(std.heap.page_allocator, 2, 1);
    // d.head(2);
}

test "get set get" {
    var a = [_]f32{ 1.2, 3.3, 5.23, 7.2 };
    const m = try DataFrame.init(std.heap.page_allocator, 2, 2, &a);
    try expect(m.get(0, 0) == @as(?f32, 1.2));
    try expect(m.get(0, 1) == @as(?f32, 3.3));
    try expect(m.get(1, 0) == @as(?f32, 5.23));
    try expect(m.get(1, 1) == @as(?f32, 7.2));
    // try expect(m.set(0, 1, @as(f32, 8.1)) == true); set() now returns void
    // try expect(m.get(0, 1) == @as(?f32, 8.1));
    defer m.deinit();
}

test "matrices dot product" {
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const am = try DataFrame.init(std.heap.page_allocator, 2, 3, &a);
    defer am.deinit();
    // am.head(10);
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const bm = try DataFrame.init(std.heap.page_allocator, 3, 2, &b);
    // bm.head(10);
    defer bm.deinit();
    const dprod = try am.dot(std.heap.page_allocator, bm);
    // dprod.head(2);
    defer dprod.deinit();
    try expect(dprod.get(0, 0) == @as(?f32, 58));
    try expect(dprod.get(0, 1) == @as(?f32, 64));
    try expect(dprod.get(1, 0) == @as(?f32, 139));
    try expect(dprod.get(1, 1) == @as(?f32, 154));
}

// test "dataframe allocation" {
//     const m = DataFrame.init(std.heap.page_allocator, 2, 1, [_]f16{ 1.2, 3.3, 5.23, 7.2 });
//     defer m.deinit();

//     try expect(m.data.?.len == 2);
//     try expect(@TypeOf(m.data.?) == []f16);
// }
