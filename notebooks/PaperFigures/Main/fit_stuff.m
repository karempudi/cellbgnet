
mu = [-2 2];
sigma = [1.0 1.0];
X = [mvnrnd(mu(1), sigma(1), 10000); mvnrnd(mu(2), sigma(2), 10000)];

GMModel = fitgmdist(X, 2);
proportions = GMModel.ComponentProportion;
figure, hold on
histogram(X, 200, "Normalization",'pdf')

x_axis = -5:0.1:5;
y_axis_1 = proportions(1) * normpdf(x_axis, GMModel.mu(1), GMModel.Sigma(1));
y_axis_2 = proportions(2) * normpdf(x_axis, GMModel.mu(2), GMModel.Sigma(2));
plot(x_axis, y_axis_1, 'Color','Red');
plot(x_axis, y_axis_2, 'Color','Green');
