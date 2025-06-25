clc; clear; close all;
rng(1);

data = readtable('Final_General_Dataset.xlsx');
X = [data.P0i, data.ggain, data.Ptank]';
Y = data.P0e';  


idx15 = find(data.ggain == 15);
idx35 = find(data.ggain == 35);
test_idx = [randsample(idx15, 10); randsample(idx35, 10)];
X_test = X(:, test_idx);
Y_test = Y(:, test_idx);

all_idx = 1:height(data);
train_val_idx = setdiff(all_idx, test_idx);
train_idx = randsample(train_val_idx, 150);
val_idx = setdiff(train_val_idx, train_idx);

X_train = X(:, train_idx);
Y_train = Y(:, train_idx);
X_val   = X(:, val_idx);
Y_val   = Y(:, val_idx);

net = feedforwardnet([5,3], 'trainlm');
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:size(X_train,2);
net.divideParam.valInd   = (size(X_train,2)+1):(size(X_train,2)+size(X_val,2));
net.divideParam.testInd  = [];
net.trainParam.epochs = 5000;
net.trainParam.goal   = 1e-3;
net.trainParam.showWindow = true;

X_total = [X_train, X_val];
Y_total = [Y_train, Y_val];

net = train(net, X_total, Y_total);

Y_val_pred = net(X_val);
val_mse = mean((Y_val - Y_val_pred).^2);
fprintf('Validation MSE (P0e): %.6f\n', val_mse);

Y_pred = net(X_test);
test_mse = mean((Y_test - Y_pred).^2);
fprintf('Test MSE (P0e): %.6f\n', test_mse);

% Grafikler
figure;

% --- Grafik 1: Validation Seti (Gerçek vs Tahmin - P0e)
subplot(1,2,1)
scatter(Y_val, Y_val_pred, 'filled')
xlabel('Gerçek P0e'); ylabel('Tahmin P0e');
title('Validation Set: Real vs Predicted');
grid on; axis equal; refline(1,0)

% --- Grafik 2: Test Seti (Gerçek vs Tahmin - P0e)
subplot(1,2,2)
scatter(Y_test, Y_pred, 'filled')
xlabel('Gerçek P0e'); ylabel('Tahmin P0e');
title('Test Set: Real vs Predicted');
grid on; axis equal; refline(1,0)


SS_res_val = sum((Y_val - Y_val_pred).^2);
SS_tot_val = sum((Y_val - mean(Y_val)).^2);
R2_val = 1 - SS_res_val / SS_tot_val;
fprintf('Validation R2: %.4f\n', R2_val);

SS_res_test = sum((Y_test - Y_pred).^2);
SS_tot_test = sum((Y_test - mean(Y_test)).^2);
R2_test = 1 - SS_res_test / SS_tot_test;
fprintf('Test R2: %.4f\n', R2_test);

% --- Residual Plot (Test Seti) ---
residuals = Y_pred - Y_test;

figure;
plot(Y_test, residuals, 'ro')
hold on;
yline(0, 'k--', 'LineWidth', 1);           % Sıfır çizgisi
yline(0.15, 'b--', 'LabelVerticalAlignment','bottom');
yline(-0.15, 'b--', 'LabelVerticalAlignment','top');
xlabel('Gerçek P0e')
ylabel('Residual (Tahmin - Gerçek)')
title('Residuals vs Gerçek P0e (Test Seti)')
grid on;


% --- P0e modeli için Training MSE ---
Y_train_pred = net(X_train);
train_mse_p0e = mean((Y_train - Y_train_pred).^2);
fprintf('Training MSE (P0e): %.6f\n', train_mse_p0e);


Y_train_pred = net(X_train);
train_mse = mean((Y_train - Y_train_pred).^2);
train_mae = mean(abs(Y_train - Y_train_pred));
SS_res_train = sum((Y_train - Y_train_pred).^2);
SS_tot_train = sum((Y_train - mean(Y_train)).^2);
train_r2 = 1 - SS_res_train / SS_tot_train;

fprintf('\n--- TRAINING ---\n');
fprintf('MSE  (P0e): %.6f\n', train_mse);
fprintf('MAE  (P0e): %.6f\n', train_mae);
fprintf('R^2  (P0e): %.4f\n', train_r2);

Y_val_pred = net(X_val);
val_mse = mean((Y_val - Y_val_pred).^2);
val_mae = mean(abs(Y_val - Y_val_pred));
SS_res_val = sum((Y_val - Y_val_pred).^2);
SS_tot_val = sum((Y_val - mean(Y_val)).^2);
val_r2 = 1 - SS_res_val / SS_tot_val;

fprintf('\n--- VALIDATION ---\n');
fprintf('MSE  (P0e): %.6f\n', val_mse);
fprintf('MAE  (P0e): %.6f\n', val_mae);
fprintf('R^2  (P0e): %.4f\n', val_r2);


Y_pred = net(X_test);
test_mse = mean((Y_test - Y_pred).^2);
test_mae = mean(abs(Y_test - Y_pred));
SS_res_test = sum((Y_test - Y_pred).^2);
SS_tot_test = sum((Y_test - mean(Y_test)).^2);
test_r2 = 1 - SS_res_test / SS_tot_test;

fprintf('\n--- TEST ---\n');
fprintf('MSE  (P0e): %.6f\n', test_mse);
fprintf('MAE  (P0e): %.6f\n', test_mae);
fprintf('R^2  (P0e): %.4f\n', test_r2);
