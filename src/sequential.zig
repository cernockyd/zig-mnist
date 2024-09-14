const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const DataFrame = @import("dataframe.zig").DataFrame;
const BatchIterator = @import("dataframe.zig").BatchIterator;
const Shape = @import("dataframe.zig").Shape;
const Data = @import("dataframe.zig").Data;
const Log = @import("log.zig").Log;

const DEBUG = false;

pub const Prediction = struct {
    a_1: DataFrame,
    z_2: DataFrame,
    a_2: DataFrame,
    z_3: DataFrame,
    a_3: DataFrame,
    z_4: DataFrame,
    output: DataFrame,

    pub fn deinit_inners(self: Prediction) void {
        self.a_1.deinit();
        self.z_2.deinit();
        self.a_2.deinit();
        self.z_3.deinit();
        self.a_3.deinit();
        self.z_4.deinit();
    }

    pub fn deinit(self: Prediction) void {
        self.deinit_inners();
        self.output.deinit();
    }
};

pub const Metrics = struct {
    cost: f128,
    accuracy: f128,

    pub fn log() void {}
    pub fn log_cost() void {}
    pub fn log_accuracy() void {}
};

pub const Gradient = struct {
    grad_1: DataFrame,
    grad_2: DataFrame,
    grad_3: DataFrame,

    pub fn deinit(self: Gradient) void {
        self.grad_1.deinit();
        self.grad_2.deinit();
        self.grad_3.deinit();
    }

    pub fn preview(self: Gradient) void {
        print("\n\nGradient 1\n----------\n", .{});
        self.grad_1.info();
        self.grad_1.head(5);

        print("\n\nGradient 2\n----------\n", .{});
        self.grad_2.info();
        self.grad_2.head(5);

        print("\n\nGradient 3\n----------\n", .{});
        self.grad_3.info();
        self.grad_3.head(5);
    }
};

pub const Sequential = struct {
    theta_1: DataFrame,
    theta_2: DataFrame,
    theta_3: DataFrame,

    pub fn deinit(self: Sequential) void {
        self.theta_1.deinit();
        self.theta_2.deinit();
        self.theta_3.deinit();
    }

    pub fn preview_thetas(self: Sequential) void {
        print("\n\nTheta 1\n-------\n", .{});
        self.theta_1.info();
        self.theta_1.head(5);
        print("\n\nTheta 2\n-------\n", .{});
        self.theta_2.info();
        self.theta_2.head(5);
        print("\n\nTheta 3\n-------\n", .{});
        self.theta_3.info();
        self.theta_3.head(5);
    }

    pub fn predict(self: Sequential, values_df: DataFrame) !Prediction {
        const a_1 = try values_df.copy();
        const z_2 = try values_df.dot(self.theta_1);
        const a_2 = try z_2.copy();
        a_2.sigmoid();
        const z_3 = try a_2.dot(self.theta_2);
        const a_3 = try z_3.copy();
        a_3.sigmoid();
        const z_4 = try a_3.dot(self.theta_3);
        const output = try z_4.copy();
        output.softmax();
        if (DEBUG) {
            print("\n\nPredictions\n-----------\n", .{});
            output.info();
        }
        // output.head(4);
        return Prediction{ .a_1 = a_1, .z_2 = z_2, .a_2 = a_2, .z_3 = z_3, .a_3 = a_3, .z_4 = z_4, .output = output };
    }

    pub fn metrics(labels: DataFrame, predictions: DataFrame) !Metrics {
        print("\n\nCost\n----\n", .{});
        var i: u32 = 0;
        var successful: usize = 0;
        var cross_entropy: f128 = 0;
        while (i < predictions.shape.m) : (i += 1) {
            const s = i * predictions.shape.n;
            const e = s + predictions.shape.n;
            const row = predictions.data.?[s..e];
            var highest: f128 = 0.0;
            var winner: usize = 0;
            for (row, 0..) |pred, order| {
                cross_entropy += if (labels.data.?[i] == @as(f32, @floatFromInt(order))) -@log(pred) else 0.0;
                if (pred > highest) {
                    highest = pred;
                    winner = order;
                }
            }
            if (labels.data.?[i] == @as(f32, @floatFromInt(winner))) {
                successful += 1;
            }
        }
        print("\nSuccessfull {d}\n", .{successful});
        predictions.info();
        const m = @as(f128, @floatFromInt(labels.shape.m));
        cross_entropy /= m;
        print("J =  {d:3} \n----\n", .{cross_entropy});
        const accuracy = @as(f128, @floatFromInt(successful)) / m * 100.0;
        print("Accuracy =  {d:3} \n----\n", .{accuracy});
        return Metrics{ .cost = cross_entropy, .accuracy = accuracy };
    }

    pub fn backprop(self: Sequential, prediction: Prediction, labels_one_hot: DataFrame) !Gradient {
        if (DEBUG) {
            print("\n\nBegin Backprop\n--------------\n", .{});
        }
        // compute error of layers
        //
        // a_err_3 = theta_3 * (g'(z_3) .* (output - y))
        // a_err_2 = theta_2 * (g'(z_2) .* a_err_3)
        // a_err_1 = theta_1 * (g'(z_1) .* a_err_2)

        const a_err_4 = try prediction.output.copy();
        if (DEBUG) {
            print("\nprediction output\n", .{});
            a_err_4.info();
            a_err_4.head(5);
            print("\nlabels_one_hot\n", .{});
            labels_one_hot.info();
            labels_one_hot.head(5);
        }
        defer a_err_4.deinit();
        try a_err_4.sub(labels_one_hot);

        if (DEBUG) {
            print("\na_err_4\n", .{});
            a_err_4.info();
            a_err_4.head(5);
        }

        const z_4_der = try prediction.z_4.copy();
        if (DEBUG) {
            print("\nz_4 copy\n", .{});
            z_4_der.info();
            z_4_der.head(100);
        }
        defer z_4_der.deinit();
        z_4_der.sigmoid_derivative();
        if (DEBUG) {
            print("\nz_4 derivative\n", .{});
            z_4_der.info();
            z_4_der.head(5);
        }
        const reverse_a_3 = try z_4_der.copy();
        if (DEBUG) {
            print("\nreverse_a_3\n", .{});
            reverse_a_3.info();
            reverse_a_3.head(5);
        }
        defer reverse_a_3.deinit();
        try reverse_a_3.mul(a_err_4);
        const theta_3_t = try self.theta_3.transpose();
        if (DEBUG) {
            print("\nreverse_a_3\n", .{});
            reverse_a_3.info();
            reverse_a_3.head(5);
        }
        defer theta_3_t.deinit();

        const a_err_3 = try reverse_a_3.dot(theta_3_t);
        defer a_err_3.deinit();
        if (DEBUG) {
            print("\na_err_3\n", .{});
            a_err_3.info();
            a_err_3.head(5);
        }

        const z_3_der = try prediction.z_3.copy();
        defer z_3_der.deinit();
        z_3_der.sigmoid_derivative();
        const reverse_a_2 = try z_3_der.copy();
        defer reverse_a_2.deinit();
        try reverse_a_2.mul(a_err_3);
        const theta_2_t = try self.theta_2.transpose();
        defer theta_2_t.deinit();

        const a_err_2 = try reverse_a_2.dot(theta_2_t);
        defer a_err_2.deinit();
        if (DEBUG) {
            print("\na_err_2\n", .{});
            a_err_2.info();
        }

        const z_2_der = try prediction.z_2.copy();
        defer z_2_der.deinit();
        z_2_der.sigmoid_derivative();
        const reverse_a_1 = try z_2_der.copy();
        defer reverse_a_1.deinit();
        try reverse_a_1.mul(a_err_2);
        const theta_1_t = try self.theta_1.transpose();
        defer theta_1_t.deinit();

        const a_err_1 = try reverse_a_1.dot(theta_1_t);
        defer a_err_1.deinit();
        if (DEBUG) {
            print("\na_err_1\n", .{});
            a_err_1.info();
        }

        // compute gradients
        // grad_(l) = a_(l) * a_err_(l+1)

        const m = @as(f32, @floatFromInt(prediction.a_1.shape.m));

        const a_err_4_t = try a_err_4.transpose();
        defer a_err_4_t.deinit();
        const grad_3 = try a_err_4_t.dot(prediction.a_3);
        if (DEBUG) {
            print("\ngrad_3\n", .{});
            grad_3.info();
            grad_3.head(40);
        }
        for (grad_3.data.?, 0..) |g, i| {
            grad_3.data.?[i] = g / m;
        }

        const a_err_3_t = try a_err_3.transpose();
        defer a_err_3_t.deinit();

        const grad_2 = try a_err_3_t.dot(prediction.a_2);
        for (grad_2.data.?, 0..) |g, i| {
            grad_2.data.?[i] = g / m;
        }

        if (DEBUG) {
            print("\ngrad_2\n", .{});
            grad_2.info();
        }

        const a_err_2_t = try a_err_2.transpose();
        defer a_err_2_t.deinit();

        const grad_1 = try a_err_2_t.dot(prediction.a_1);
        for (grad_1.data.?, 0..) |g, i| {
            grad_1.data.?[i] = g / m;
        }

        if (DEBUG) {
            print("\ngrad_1\n", .{});
            grad_1.info();
        }

        try grad_1.clip();
        try grad_2.clip();
        try grad_3.clip();

        return Gradient{ .grad_1 = grad_1, .grad_2 = grad_2, .grad_3 = grad_3 };
    }

    pub fn update(self: Sequential, gradient: Gradient, epoch: usize) !void {
        _ = epoch;
        //  print("\n\nBegin Update\n--------------\n", .{});
        const alpha: f32 = 0.01;
        const grad_1 = try gradient.grad_1.transpose();
        defer grad_1.deinit();
        const grad_2 = try gradient.grad_2.transpose();
        defer grad_2.deinit();
        const grad_3 = try gradient.grad_3.transpose();
        defer grad_3.deinit();
        // try grad_1.clip();
        // try grad_2.clip();
        // try grad_3.clip();
        // grad_3.head(400);

        for (grad_1.data.?, 0..) |g, i| {
            grad_1.data.?[i] = g * alpha;
        }

        for (grad_2.data.?, 0..) |g, i| {
            grad_2.data.?[i] = g * alpha;
        }

        for (grad_3.data.?, 0..) |g, i| {
            grad_3.data.?[i] = g * alpha;
        }

        try self.theta_1.sub(grad_1);
        try self.theta_2.sub(grad_2);
        try self.theta_3.sub(grad_3);
    }

    pub fn train(self: Sequential, allocator: Allocator, train_data: Data, test_data: Data) !void {
        print("\n\nBegin training\n--------------\n", .{});
        const epochs: u32 = 300;
        // compute cost
        // compute gradients using backpropagation
        // update weights
        const model = Sequential{ .theta_1 = self.theta_1, .theta_2 = self.theta_2, .theta_3 = self.theta_3 };
        const labels_one_hot = try train_data.labels_df.one_hot(10);
        defer labels_one_hot.deinit();

        const cost_log = try Log.init("./src/control/data-cost.txt");
        defer cost_log.deinit();
        const train_accuracy_log = try Log.init("./src/control/data-train-accuracy.txt");
        defer train_accuracy_log.deinit();
        const test_accuracy_log = try Log.init("./src/control/data-test-accuracy.txt");
        defer test_accuracy_log.deinit();

        var batch_iter = BatchIterator{
            .index = 0,
            .batch_size = 10,
            .data = Data{
                .values_df = train_data.values_df,
                .labels_df = labels_one_hot,
            },
        };

        for (0..epochs) |epoch| {
            // model.var batch_count: u32 = 0;
            while (batch_iter.next()) |batch| {
                const percentage = (@as(f32, @floatFromInt(batch_iter.index)) / @as(f32, @floatFromInt(train_data.values_df.shape.m))) * 100.0;
                print("Epoch #{} ... progress [{d:.1} %]\r", .{ epoch, percentage });
                // batch.labels_df.info();
                // batch.values_df.info();
                const prediction = try model.predict(batch.values_df);
                defer prediction.deinit();
                const gradient = try model.backprop(prediction, batch.labels_df);
                defer gradient.deinit();

                try model.update(gradient, epoch);
            }
            batch_iter.index = 0;
            const prediction = try model.predict(train_data.values_df);
            defer prediction.deinit();
            const model_metrics = try Sequential.metrics(train_data.labels_df, prediction.output); // should be about -+ -log(1/10)
            const cost_line = try std.fmt.allocPrint(
                allocator,
                "{},{d}\n",
                .{ epoch, model_metrics.cost },
            );
            try cost_log.write(cost_line);
            defer allocator.free(cost_line);
            const accuracy_line = try std.fmt.allocPrint(
                allocator,
                "{},{d}\n",
                .{ epoch, model_metrics.accuracy },
            );
            try train_accuracy_log.write(accuracy_line);
            if (model_metrics.accuracy > 98) {
                print("Training reached 98 % accuracy at epoch # {}. Stopping", .{epoch});
                break;
            }
            defer allocator.free(accuracy_line);

            // test accuracy
            const test_prediction = try model.predict(test_data.values_df);
            defer test_prediction.deinit();
            const test_metrics = try Sequential.metrics(test_data.labels_df, test_prediction.output); // should be about -+ -log(1/10)
            const test_accuracy_line = try std.fmt.allocPrint(
                allocator,
                "{},{d}\n",
                .{ epoch, test_metrics.accuracy },
            );
            try test_accuracy_log.write(test_accuracy_line);
            defer allocator.free(test_accuracy_line);
        }
    }
};
