clc; clear; close all;
rng(1);  % Reproducibility

% --- Veriyi oku
data = readtable('Final_General_Dataset.xlsx');

% --- Girişler
X = [data.P0i, data.ggain, data.Ptank]';  % 3 x 240

% --- Hedef (P0e)
Y = data.P0e';                            % 1 x 240

% --- Test seti belirle (ggain = 15 ve 35'ten 10'ar örnek)
idx15 = find(data.ggain == 15);
idx35 = find(data.ggain == 35);
test_idx = [randsample(idx15, 10); randsample(idx35, 10)];

X_test = X(:, test_idx);
Y_test = Y(:, test_idx);

% --- Geriye kalan veriler: training + validation
all_idx = 1:height(data);
train_val_idx = setdiff(all_idx, test_idx);
X_tv = X(:, train_val_idx);
Y_tv = Y(:, train_val_idx);

% --- Cross-validation ayarları
k = 11;
cv = cvpartition(length(train_val_idx), 'KFold', k);

% --- Performans metrikleri
mse_train = zeros(1, k);
mse_val   = zeros(1, k);
r2_train  = zeros(1, k);
r2_val    = zeros(1, k);
mae_train = zeros(1, k);  % 👈 MAE için alan açıldı
mae_val   = zeros(1, k);
nets = cell(1, k);

for i = 1:k
    train_i = training(cv, i);
    val_i   = test(cv, i);

    X_train = X_tv(:, train_i);
    Y_train = Y_tv(:, train_i);
    X_val   = X_tv(:, val_i);
    Y_val   = Y_tv(:, val_i);

    % --- Ağ tanımı (6 nöron, 1 gizli katman)
    net = feedforwardnet(6, 'trainlm');
    net.trainParam.showWindow = false;
    net.divideFcn = 'dividetrain';

    % --- Eğitim
    net = train(net, X_train, Y_train);

    % --- Tahminler
    Y_train_pred = net(X_train);
    Y_val_pred   = net(X_val);

    % --- Performans hesaplamaları
    mse_train(i) = mean((Y_train - Y_train_pred).^2);
    mse_val(i)   = mean((Y_val - Y_val_pred).^2);

    SS_res_train = sum((Y_train - Y_train_pred).^2);
    SS_tot_train = sum((Y_train - mean(Y_train)).^2);
    r2_train(i) = 1 - SS_res_train / SS_tot_train;

    SS_res_val = sum((Y_val - Y_val_pred).^2);
    SS_tot_val = sum((Y_val - mean(Y_val)).^2);
    r2_val(i) = 1 - SS_res_val / SS_tot_val;

    % --- MAE hesapla
    mae_train(i) = mean(abs(Y_train - Y_train_pred));
    mae_val(i)   = mean(abs(Y_val   - Y_val_pred));

    nets{i} = net;
end

% --- En iyi ağı seç
[~, best_i] = min(mse_val);
best_net = nets{best_i};

% --- Test tahminleri
Y_pred_test = best_net(X_test);

% --- Test metrikleri
mse_test = mean((Y_test - Y_pred_test).^2);
mae_test = mean(abs(Y_test - Y_pred_test));  % 👈 MAE eklendi
SS_res_test = sum((Y_test - Y_pred_test).^2);
SS_tot_test = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - SS_res_test / SS_tot_test;

% --- Performans çıktısı
fprintf("Performans Sonuçları (P0e - en iyi ağ Fold #%d):\n", best_i);
fprintf("  Training:   MSE = %.4f, R² = %.4f, MAE = %.4f\n", ...
    mse_train(best_i), r2_train(best_i), mae_train(best_i));
fprintf("  Validation: MSE = %.4f, R² = %.4f, MAE = %.4f\n", ...
    mse_val(best_i), r2_val(best_i), mae_val(best_i));
fprintf("  Test:       MSE = %.4f, R² = %.4f, MAE = %.4f\n", ...
    mse_test, r2_test, mae_test);

% --- Grafik: Validation set (Real vs Predicted)
val_i = test(cv, best_i);
Y_val_real = Y_tv(:, val_i);
Y_val_pred = best_net(X_tv(:, val_i));

figure;
subplot(1,2,1)
scatter(Y_val_real, Y_val_pred, 'filled')
xlabel('Gerçek P0e'); ylabel('Tahmin P0e');
title('Validation Set: Real vs Predicted'); grid on; refline(1,0)

% --- Grafik: Test set (Real vs Predicted)
subplot(1,2,2)
scatter(Y_test, Y_pred_test, 'filled')
xlabel('Gerçek P0e'); ylabel('Tahmin P0e');
title('Test Set: Real vs Predicted'); grid on; refline(1,0)

% --- Grafik: Residuals vs Real (Test set)
residuals = Y_test - Y_pred_test;
figure;
scatter(Y_test, residuals, 'filled')
yline(0.15, 'b--', 'LabelVerticalAlignment','bottom');
yline(-0.15, 'b--', 'LabelVerticalAlignment','top');
xlabel('Gerçek P0e'); ylabel('Residual (Gerçek - Tahmin)');
title('Test Set: Residuals vs Gerçek Değerler'); grid on;
yline(0, '--r');

fprintf("\nTüm fold’ların ortalama sonuçları:\n");
fprintf("  Avg Training:   MSE = %.4f, R² = %.4f, MAE = %.4f\n", ...
    mean(mse_train), mean(r2_train), mean(mae_train));
fprintf("  Avg Validation: MSE = %.4f, R² = %.4f, MAE = %.4f\n", ...
    mean(mse_val), mean(r2_val), mean(mae_val));

