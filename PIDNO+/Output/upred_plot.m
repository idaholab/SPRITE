% load ODE_Preds;
idx = 1:2;
idx = [1,3,5,8,9];
idx = [1,3,8,16,19];
idx = [1,2,4,9,10]; % test5
idx = [1,2,6,8,15,16]; % test6
idx = 1:5;

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
ylim([0 1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% idx = 1:5;
% screenN = 100;
% x_test1 = x_test(1:screenN,:);
% u_test = Pmass_test(:,1:screenN);
% u_pred = Pmass_pred(:,1:screenN);
% f_test = Fmass_test(:,1:screenN);
% 
% figure
% hold on;
% plot(x_test1, u_test(idx, :), 'k-', 'linewidth', 2.0);
% plot(x_test1, u_pred(idx, :), 'r--', 'linewidth', 1.5);
% plot(x_test1, f_test(idx, :), 'b--', 'linewidth', 1.5);
% xlabel('$t/s$', 'fontsize', 10, 'interpreter', 'latex');
% ylabel('$F$', 'fontsize', 10, 'interpreter', 'latex');
% 
