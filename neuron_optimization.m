clc; clear; close all;
rng(1);  % Reproducibility

% --- Veriyi oku
data = readtable('Final_General_Dataset.xlsx');

% --- Girişler
X = [data.P0i, data.ggain, data.Ptank]';

% --- Hedefler
Y_ipas = data.IPAS';
Y_p0e  = data.P0e';

% --- Test seti
idx15 = find(data.ggain == 15);
idx35 = find(data.ggain == 35);
test_idx = [randsample(idx15, 10); randsample(idx35, 10)];

X_test = X(:, test_idx);
Y_ipas_test = Y_ipas(:, test_idx);
Y_p0e_test  = Y_p0e(:, test_idx);

% --- Eğitim + validasyon seti
train_val_idx = setdiff(1:height(data), test_idx);
X_tv = X(:, train_val_idx);
Y_ipas_tv = Y_ipas(:, train_val_idx);
Y_p0e_tv  = Y_p0e(:, train_val_idx);

% --- Cross-validation
k = 5;
cv = cvpartition(length(train_val_idx), 'KFold', k);

% --- Nöron sayısı aralığı
neuron_range = 2:15;
n_trials = length(neuron_range);

% --- Sonuçları sakla
mae_val_ipas = zeros(1, n_trials);
mae_val_p0e  = zeros(1, n_trials);

for n = 1:n_trials
    h = neuron_range(n);
    mae_val_folds_ipas = zeros(1, k);
    mae_val_folds_p0e  = zeros(1, k);

    for i = 1:k
        train_i = training(cv, i);
        val_i   = test(cv, i);

        % --- IPAS modeli
        net_ipas = feedforwardnet(h, 'trainlm');
        net_ipas.trainParam.showWindow = false;
        net_ipas.divideFcn = 'dividetrain';
        net_ipas = train(net_ipas, X_tv(:, train_i), Y_ipas_tv(:, train_i));
        Y_pred_ipas_val = net_ipas(X_tv(:, val_i));
        mae_val_folds_ipas(i) = mean(abs(Y_ipas_tv(:, val_i) - Y_pred_ipas_val));

        % --- P0e modeli
        net_p0e = feedforwardnet(h, 'trainlm');
        net_p0e.trainParam.showWindow = false;
        net_p0e.divideFcn = 'dividetrain';
        net_p0e = train(net_p0e, X_tv(:, train_i), Y_p0e_tv(:, train_i));
        Y_pred_p0e_val = net_p0e(X_tv(:, val_i));
        mae_val_folds_p0e(i) = mean(abs(Y_p0e_tv(:, val_i) - Y_pred_p0e_val));
    end

    % Ortalama validation hataları
    mae_val_ipas(n) = mean(mae_val_folds_ipas);
    mae_val_p0e(n)  = mean(mae_val_folds_p0e);

    fprintf("Nöron: %2d | IPAS MAE = %.4f | P0e MAE = %.4f\n", ...
            h, mae_val_ipas(n), mae_val_p0e(n));
end

% --- En iyi nöron sayısını seç
[~, best_idx_ipas] = min(mae_val_ipas);
[~, best_idx_p0e]  = min(mae_val_p0e);

fprintf("\n👉 IPAS için en iyi nöron sayısı: %d (MAE = %.4f)\n", ...
        neuron_range(best_idx_ipas), mae_val_ipas(best_idx_ipas));
fprintf("👉 P0e  için en iyi nöron sayısı: %d (MAE = %.4f)\n", ...
        neuron_range(best_idx_p0e), mae_val_p0e(best_idx_p0e));

% --- Grafiksel gösterim
figure;
plot(neuron_range, mae_val_ipas, '-o', 'LineWidth', 2); hold on;
plot(neuron_range, mae_val_p0e, '-s', 'LineWidth', 2);
xlabel('Gizli Katmandaki Nöron Sayısı');
ylabel('Ortalama Validation MAE');
legend('IPAS', 'P0e', 'Location', 'best');
title('Hidden Layer Nöron Sayısı Optimizasyonu');
grid on;
