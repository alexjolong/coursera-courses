function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

[m, n] = size(X)
[num_labels,a_t_2] = size(all_theta) %k, n+1


% You need to return the following variables correctly 
p = zeros(m, 1);

% Add ones to the X data matrix so that X is m,n+1
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% m = number of examples, n = number of features, k = number of classes
% all_theta is kX(n + 1)
% X is mX(n + 1)
% p is mX1
% we want p to be a vector where every row is the classification for that
% row from X.
% we calculate the classification for each example i by making a new 
% vector temp = (kX1) for each example, where the value at row j is the 
% probability that i is in class j.
% then we set p(i) to be the max of vector temp, and repeat for every i in
% 1:m
temp = zeros(m,num_labels);
for i=1:num_labels
    temp(:,i) = X * all_theta(i,:)'; % the result is a mX(n+1) vector
end

[~, p] = max(temp, [], 2)

% =========================================================================


end
