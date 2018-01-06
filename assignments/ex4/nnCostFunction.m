function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
% Theta1 now has size 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Theta2 now has size 10 x 26

% Setup some useful variables
m = size(X, 1);
K = num_labels;
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1), X]; % add our bias at a1(0) for all training examples
% Theta1 and Theta1_grad are 25x401
% Theta2 and Theta2_grad are 10x26

for i = 1:m
    a1 = X(i,:)'; % a1 is our input for one example. a 401 x 1 vector
    z2 = Theta1 * a1; % result is 25 x 1
    a2 = [1; sigmoid(z2)]; % added bias at a2(0)
    % we need to add a bias value to Theta2
    z3 = Theta2 * a2; % result is 10 x 1
    a3 = sigmoid(z3);

    % newY is 10 x 1, with a 1 in the slot corresponding to the digit given
    % by y(i) (which is the digit shown in x(i,:))
    newY = zeros(K, 1);
    newY(y(i)) = 1; 
    
    % delta3 is a 10 x 1 vector which is the error between our estimate and
    % the actual answer for this training example.
    delta3 = (a3 - newY); 
    
    % we don't want to update the bias in our backpropogation, so we drop
    % the first element of the first term (which corresponds to the first
    % row of Theta2' -- our output layer bias weights.
    % delta2 is 26x10 * 10x1 = 26x1 .* 26x1 = 26x1
    % we add a bias to keep the dimensions consistent when multiplying, but
    % we won't use it for calculations later.
    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
    delta2 = delta2(2:end);
    % there is no delta1, because the first layer should have no error
    
    % Compute the cost J
    term1 = (-newY) .* log(a3);
    term2 = (1 - newY) .* log(1 - a3);
    innerSum = sum(term1 - term2);    
    J = J + innerSum;
    
    % accumulate Delta (the partial derivative of the cost function with
    % respect to Theta1 and Theta 2
    % Theta2_grad is 10x1 * 1x26 = 10 x 26
    % we want Theta2_grad to be 10x26
    Theta2_grad = Theta2_grad + (delta3 * a2');
    
    % Theta1_grad is 25 x 1 * 1 x 401 = 25 x 401
    % we want Theta1_grad to be 25x401
    Theta1_grad = Theta1_grad + (delta2 * a1');
    
end

regTerm1 = sum(sum(Theta1(:, 2:end).^2));
regTerm2 = sum(sum(Theta2(:, 2:end).^2));
reg = lambda * (regTerm1 + regTerm2) / (2 * m);

J = (J / m) + reg;

regTheta1 = Theta1;
regTheta2 = Theta2;
regTheta1(:, 1) = zeros(size(Theta1,1),1);
regTheta2(:, 1) = zeros(size(Theta2,1),1);
Theta1_grad = (Theta1_grad / m) + ((lambda/m) * regTheta1);
Theta2_grad = (Theta2_grad / m) + ((lambda/m) * regTheta2);

% -------------------------------------------------------------


for t=1:m
    
    
end



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
