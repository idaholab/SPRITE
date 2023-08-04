% load ODE_Preds;
idx = [2,3,4,7,9]; % v1 0.3k
idx = 1:100;

x_test = x_test;
u_test = u_test;
u_pred = u_pred;

figure
hold on;
plot(x_test, u_test(idx, :), 'k-', 'linewidth', 1.0);
plot(x_test, u_pred(idx, :), 'r-.', 'linewidth', 2.0);
xlabel('Sieve Size (mm)', 'fontsize', 20);
ylabel('Cumulative Mass %', 'fontsize', 20);
% lgd = legend('test','pred');
% lgd.FontSize = 20;
xlim([0 0.4])
ylim([0 1.01])
