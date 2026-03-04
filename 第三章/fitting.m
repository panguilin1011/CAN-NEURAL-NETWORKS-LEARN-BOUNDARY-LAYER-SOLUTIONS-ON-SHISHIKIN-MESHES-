clear
format long
%参数
epsilon = 1e-6;
N = 32;
N_iter = 1e5;
eta = 5e-5;

load('myRandState.mat'); % 加载随机状态
rng(s); 

%分界点
tau = Tau(epsilon,N);
%2个h
h = 2*(1-tau)/N; %(0,1-tau)上的步长
H = 2*tau/N;        %(tau,1)上的步长
x(1:N/2+1) = linspace(0,1-tau,N/2+1);
x(N/2+1:N+1) = linspace(1-tau,1,N/2+1);

%真解
u_exact = @(x) (-exp(-1/epsilon)+exp((x-1)./epsilon))/(exp(-1/epsilon)-1) + x;
u_e = u_exact(x);


F_value = cell(1,N+1);
M = cell(1,N+1);
sigma = cell(1, N+1);
sigma_der = cell(1, N+1);
b = rand(N+1, 1);
W1 = rand(N+1, 1);
W2 = rand(1,N+1);

for cycle = 1:N_iter
    db = zeros(N+1,1);
    dW1 = zeros(N+1,1);
    dW2 = zeros(1,N+1);
    for j = 1:N+1
        F_value{j} = F(x(j),W2,W1,b);
    end
    
    for j = 1:N+1
        M{j} = u_e(j) - F_value{j};
    end

    for j = 1:N+1
        z{j} = W1*x(j) + b;
        sigma{j} = relu(z{j});
        sigma_der{j} = relu_der(z{j});
    end

    for j = 1:N+1
        db = db + M{j}.*(-W2'.*sigma_der{j});
    end
    db = (2/(N+1))*db;

    for j = 1:N+1
        dW1 = dW1 + M{j}*(-W2'*x(j).*sigma_der{j});
    end
    dW1 = (2/(N+1))*dW1;

    for j = 1:N+1
        dW2 = dW2 + (M{j}*(-sigma{j}))';
    end
    dW2 = (2/(N+1))*dW2;

    b = b - eta*db;
    W1 = W1 - eta*dW1;
    W2 = W2 - eta*dW2;

    newcost = cost(x,N,W2,W1,b,u_e);
    cost_value(cycle) = newcost;

    %L2
    diff = abs(u_e - cell2mat(F_value));
    L_2(cycle) = sqrt(sum(diff.^2)*h);
end

F_values = cell2mat(F_value);
error = abs(u_e-F_values);
figure(1)
h1 = plot(x,u_e,'r-','LineWidth',1.5); hold on;

for i = 1:length(x)
    xline(x(i), ':k', 'HandleVisibility','off'); % 图例忽略这些线
end

h2 = plot(x,F_values,'b--','LineWidth',1.5);
hold off

legend([h1 h2], {'精确解','NN'}, 'Location','northwest');


figure(2)
yyaxis left;
semilogy(1:N_iter,cost_value,'r-','LineWidth',1.5);
xlabel('迭代步数');
ylabel('损失函数值');
yyaxis right;
semilogy(1:N_iter,L_2,'b-','LineWidth',1.5);
ylabel('L^2误差');
legend('Cost','L^2误差','Location','northeast');

figure(3)
plot(x,error,'LineWidth', 1.5);
legend('节点绝对值误差','Location','northwest');
cost_value(end)