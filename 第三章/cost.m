function costvalue = cost(x,N,W2,W1,b,u_e)
costvalue = 0;
    for i = 1:N+1
        costvalue = costvalue + (u_e(i) - W2*relu(W1*x(i)+b)).^2;
    end
costvalue = (1/(N+1))*costvalue;
