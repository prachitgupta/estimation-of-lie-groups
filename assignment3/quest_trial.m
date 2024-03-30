function [C,omega,q,Z] = quest_trial(v_i, v_b, w)

    % Compute B matrix (Eqn.38)
    B = (v_b .* repmat(w, [1, 3])') * v_i';

    % Compute sigma (Eqn.44)
    sigma = trace(B);

    % Compute S matrix (Eqn.45)
    S = B + B';

    % Z
    Z = [B(2, 3) - B(3, 2); B(3, 1) - B(1, 3); B(1, 2) - B(2, 1)];
    kappa = trace(det(S)*inv(S));

    % Compute delta and kappa (Eqn.63)
    delta = det(S);

    % Compute coefficients for lambda (Eqn.71)
    a = sigma^2 - kappa;
    b = sigma^2 + Z' * Z;
    c = delta + Z' * S * Z;
    d = Z' * S^2 * Z;
    constant = a * b + c * sigma - d;

    % Initialize lambda
    lambda = sum(w);

    %first guess as 1 as men% Define tolerance for iterative solution
    tolerance = 10e-5;
    last_lambda = 0.0;
    % Using newton rahpson as mentioned in paper. instead of eigen value. Iterate to find lambda
    % Standard newton rahpson method
    while abs(lambda - last_lambda) >= tolerance
        last_lambda = lambda;

        % Update lambda using Newton-Raphson method
        f = lambda^4 - (a + b) * lambda^2 - c * lambda + constant;
        f_dot = 4 * lambda^3 - 2 * (a + b) * lambda - c;
        lambda = lambda - f / f_dot;
    end

    % Compute omega, alpha, beta, and gamma (Eqn.66)
    % eqn 67 tells to assume omega=lambda_max
    omega = lambda;
    display(omega)
    alpha = omega^2 - sigma^2 + kappa;
    beta = omega - sigma;
    gamma = (omega + sigma) * alpha - delta;

    % (Eqn.68) Compute X vector 
    X = (alpha * eye(3) + beta * S + S^2) * Z;

    % Compute the optimal quaternion (Eqn.69)
    q = [X; gamma] ./ sqrt(gamma^2 + norm(X)^2);

    % Convert the optimal quaternion to DCM
    % standard method
    q0 = q(4);
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    
    C(1,1) = q0^2 + q1^2 - q2^2 - q3^2;
    C(1,2) = 2*(q1*q2 + q0*q3);
    C(1,3) = 2*(q1*q3 - q0*q2);
    C(2,1) = 2*(q1*q2 - q0*q3);
    C(2,2) = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    C(2,3) = 2*(q2*q3 + q0*q1);
    C(3,1) = 2*(q1*q3 + q0*q2);
    C(3,2) = 2*(q2*q3 - q0*q1);
    C(3,3) = q0^2 - q1^2 - q2^2 + q3^2;
end


