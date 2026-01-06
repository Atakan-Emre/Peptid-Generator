function [bestTour, bestCost, info] = ils_tsp(problem, opts, seed)
%ILS_TSP Iterated Local Search (double-bridge perturbation + 2-opt).
%
% Very strong baseline for Euclidean TSP and usually beats a plain SA quickly.
%
%   [tour,cost,info] = ils_tsp(problem, opts, seed)
%
% opts (typical):
%   .maxIter      = 1000;
%   .twoOptPasses = 5;      % local search intensity
%   .kickStrength = 1;      % number of double-bridge moves per iter (>=1)
%   .initMethod   = 'nn';   % 'nn' or 'rand'
%
% Minimize tour length.

    if nargin < 3, seed = 1; end
    rng(seed);

    dflt = struct('maxIter',1000,'twoOptPasses',5,'kickStrength',1,'initMethod','nn');
    opts = set_defaults(opts, dflt);

    n = problem.n;

    % init
    switch lower(opts.initMethod)
        case 'nn'
            tour = tsp_ops('nn', problem.D);
        case 'rand'
            tour = tsp_ops('rand', n);
        otherwise
            error('Unknown initMethod: %s', opts.initMethod);
    end

    [tour, cost] = tsp_twoopt_ls(problem, tour, opts.twoOptPasses);
    bestTour = tour; bestCost = cost;

    curveBest = nan(opts.maxIter,1);
    t0 = tic;

    for k = 1:opts.maxIter
        cand = tour;

        % perturb ("kick")
        for kk = 1:max(1,opts.kickStrength)
            cand = double_bridge(cand);
        end

        % local search
        [cand, candCost] = tsp_twoopt_ls(problem, cand, opts.twoOptPasses);

        % accept if better than current
        if candCost < cost
            tour = cand;
            cost = candCost;
        end

        if cost < bestCost
            bestCost = cost;
            bestTour = tour;
        end

        curveBest(k) = bestCost;
    end

    info = struct();
    info.curveBest = curveBest;
    info.timeSec = toc(t0);
end

% ---------- helpers ----------
function tour2 = double_bridge(tour)
    n = numel(tour);
    if n < 8
        tour2 = tsp_ops('twoopt', tour); % fallback
        return;
    end

    cuts = sort(randperm(n-1, 4)); % 4 cut points in 1..n-1
    a = cuts(1); b = cuts(2); c = cuts(3); d = cuts(4);

    p1 = tour(1:a);
    p2 = tour(a+1:b);
    p3 = tour(b+1:c);
    p4 = tour(c+1:d);
    p5 = tour(d+1:end);

    % classic double-bridge: 1 | 3 | 2 | 4 (keep 5 at end)
    tour2 = [p1, p3, p2, p4, p5];
end

function s = set_defaults(s, d)
    if isempty(s), s = struct(); end
    f = fieldnames(d);
    for i=1:numel(f)
        if ~isfield(s,f{i}) || isempty(s.(f{i}))
            s.(f{i}) = d.(f{i});
        end
    end
end
