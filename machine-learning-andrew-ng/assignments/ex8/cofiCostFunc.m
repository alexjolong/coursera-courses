function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.

% Y is nm x nu, where the value of each element is a rating 1-5
% R is nm x nu, where the value of each element is a binary value
% representing whether or not user j has rated movie i (R(i,j)==1) or not
% (R(i,j)==0)

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% X is a nm x 100 matrix, where each row i is a movie and each column j is
% a feature for that movie
% Theta is a nu x 100 matrix, where each row i is a user and each column
% is a feature that user
            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X)); % a vector where each element is the gradient for a movie
Theta_grad = zeros(size(Theta)); % a vector where every elementn is the gradient for a user

% the model predicts the rating for movie i by user j as y(i,j)=theta(j)' *
% x(i)


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

Predicts = X * Theta'; % this should be a nm X nu matrix, same as Y
Predicts_diff = Predicts - Y; 
Squared_diff = (Predicts_diff .^ 2) / 2;
Squared_diff_cleaned = Squared_diff .* R;
J = sum(sum(Squared_diff_cleaned));

reg_term_1 = (lambda / 2) * sum(sum(Theta .^ 2));
reg_term_2 = (lambda / 2) * sum(sum(X .^ 2));
J = J + reg_term_1 + reg_term_2;

% gradients:
% X_grad is nmX100
% Theta_Grad is nuX100

for i=1:num_movies
    idx = find(R(i,:)==1);
    Theta_temp = Theta(idx,:); %num_cleaned_users X 100
    Y_temp = Y(i,idx); % 1xnum_cleaned_users
    %X(i,:) is 1X100
    
    x_grad_for_i_j = ((X(i,:)*Theta_temp') - Y_temp) * Theta_temp;
    X_grad(i,:) = x_grad_for_i_j + (lambda * X(i,:));
end
for j=1:num_users
    idx = find(R(:,j)==1);
    X_temp = X(idx,:);  %num_cleaned_movies x 100
    Y_temp = Y(idx,j); %num_moviesx1
    %Theta(j,:) is 1X100
    
    theta_grad_for_j_i=((X_temp * Theta(j,:)') - Y_temp)' * X_temp;
    Theta_grad(j,:) = theta_grad_for_j_i + (lambda * Theta(j,:));
end

%{
% we could also do it with loops
for i=1:num_movies
    for j=1:num_users
        if (R(i,j) == 1)
            J = J + ((Theta(j,:) * X(i,:)') - (Y(i,j))) ^ 2;      
        end
    end
end

J = J / 2;
%}






% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
