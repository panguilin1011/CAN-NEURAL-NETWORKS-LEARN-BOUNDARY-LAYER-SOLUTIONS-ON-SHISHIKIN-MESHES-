function F_value = F(x,W2,W1,b)
    F_value = W2*relu(W1*x+b);
end