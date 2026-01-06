function [bestTour, bestCost, info] = moco_cem_tsp(problem, opts, seed)
%MOCO_CEM_TSP Improved MOCO-inspired heatmap/meta-optimizer for TSP (MATLAB).
%
% This is a practical MOCO-style loop:
%   heatmap theta -> sample tours -> update theta from elites
% using a Cross-Entropy / ACO-style update.
%
% Further improvements added in this version:
% - Heuristic bias (distance) with weight betaHeur (ACO-like)
% - Candidate list sampling (nearest-neighbor restriction) for better tours + speed
% - Elitist reinforcement of the incumbent best tour (bestReinf)
% - Simple "reheat" when stagnating: temporarily more exploration (tau/baseMix) and gentler updates
% - Optional probability upper bound (maxProb) + floor (minProb) for stability (MMAS-like)
%
% Minimization: tour length (smaller cost is better).
%
% opts (all optional, sensible defaults):
%   % Budget
%   .maxIter           (default 200)
%   .batchSize         (default 96)
%   .eliteFrac         (default 0.12)
%   .alpha             (default 0.18)    % EMA update strength in probability space
%
%   % Sampling temperature schedule
%   .tauStart          (default 1.80)
%   .tauEnd            (default 0.65)
%   .tauSchedule       (default 'linear') % 'linear' | 'exp'
%
%   % Heuristic + candidate list
%   .betaHeur          (default 1.0)     % weight of distance heuristic in sampling
%   .useCandidateList  (default true)
%   .candListSize      (default 20)      % NN candidates per node (<= n-1)
%
%   % Local search
%   .twoOptMode        (default 'elite') % 'none' | 'elite' | 'all'
%   .twoOptEliteFrac   (default 0.25)
%   .twoOptPasses      (default 2)
%   .twoOptOnBest      (default true)
%
%   % Update shaping
%   .eliteWeighting    (default 'rank')  % 'rank' | 'exp' | 'uniform'
%   .weightGamma       (default 0.20)    % for 'exp'
%   .bestReinf         (default 0.25)    % add best tour edges into update
%   .baseMix           (default 0.05)    % mix base distribution into update (avoid collapse)
%
%   % Probability bounds (MMAS-like)
%   .minProb           (default 1e-6)
%   .maxProb           (default 0.95)    % <=1, applied before row-normalization
%
%   % Stagnation reheat
%   .stagnationIters   (default 35)      % if no improvement for this many iters => reheat
%   .reheatFor         (default 10)      % duration (iters)
%   .tauBoost          (default 1.35)
%   .baseMixBoost      (default 0.08)
%   .alphaDrop         (default 0.70)    % alphaEff = alpha * alphaDrop during reheat
%
% Output:
%   bestTour, bestCost, info.curveBest, info.timeSec

    if nargin < 3 || isempty(seed), seed = 1; end
    rng(seed);

    if nargin < 2, opts = struct(); end

    n = problem.n;
    D = problem.D;

    dflt = struct( ...
        'maxIter',         200, ...
        'batchSize',       96, ...
        'eliteFrac',       0.12, ...
        'alpha',           0.18, ...
        'tauStart',        1.80, ...
        'tauEnd',          0.65, ...
        'tauSchedule',     'linear', ...
        'betaHeur',        1.0, ...
        'useCandidateList', true, ...
        'candListSize',    20, ...
        'twoOptMode',      'elite', ...
        'twoOptEliteFrac', 0.25, ...
        'twoOptPasses',    2, ...
        'twoOptOnBest',    true, ...
        'eliteWeighting',  'rank', ...
        'weightGamma',     0.20, ...
        'bestReinf',       0.25, ...
        'baseMix',         0.05, ...
        'minProb',         1e-6, ...
        'maxProb',         0.95, ...
        'stagnationIters', 35, ...
        'reheatFor',       10, ...
        'tauBoost',        1.35, ...
        'baseMixBoost',    0.08, ...
        'alphaDrop',       0.70 ...
    );
    opts = set_defaults(opts, dflt);

    % --------------------
    % Precompute heuristic term and candidate lists
    % --------------------
    % Heuristic eta(i,j) ∝ 1/(D+eps). We'll use log(eta) in scoring.
    etaLog = -log(D + 1e-12);
    etaLog(1:n+1:end) = -inf;

    candK = min(max(2, round(opts.candListSize)), n-1);
    candList = cell(n,1);
    if opts.useCandidateList
        for i = 1:n
            [~, ord] = sort(D(i,:), 'ascend');
            ord(ord == i) = [];
            candList{i} = ord(1:candK);
        end
    else
        for i = 1:n
            candList{i} = [];
        end
    end

    % --------------------
    % Base distribution (heuristic)
    % --------------------
    baseP = build_baseP(D);
    theta = log(baseP + 1e-12);
    theta(1:n+1:end) = -inf;

    % --------------------
    % Initial incumbent
    % --------------------
    bestTour = tsp_ops('nn', D);
    [bestTour, bestCost] = tsp_twoopt_ls(problem, bestTour, opts.twoOptPasses);

    curveBest = nan(opts.maxIter,1);
    t0 = tic;

    eliteCount = max(1, round(opts.batchSize * opts.eliteFrac));
    twoOptEliteCount = max(1, round(opts.batchSize * opts.twoOptEliteFrac));

    lastImproveIter = 0;

    for it = 1:opts.maxIter
        % -------- reheat logic --------
        noImp = it - lastImproveIter;
        inReheat = (noImp >= opts.stagnationIters) && (noImp < opts.stagnationIters + opts.reheatFor);

        tau = get_tau(opts, it);
        alphaEff = opts.alpha;
        baseMixEff = opts.baseMix;

        if inReheat
            tau = tau * opts.tauBoost;
            alphaEff = max(0.05, opts.alpha * opts.alphaDrop);
            baseMixEff = min(0.30, opts.baseMix + opts.baseMixBoost);
        end

        tours = zeros(opts.batchSize, n);
        costs = inf(opts.batchSize, 1);

        % --------------------
        % Sample a batch
        % --------------------
        for b = 1:opts.batchSize
            start = randi(n);
            tour  = sample_tour(theta, etaLog, candList, start, tau, opts.betaHeur);

            tours(b,:) = tour;
            costs(b)   = tsp_fitness(tour, problem);
        end

        % --------------------
        % 2-opt refinement (elite-only or all)
        % --------------------
        switch lower(opts.twoOptMode)
            case 'all'
                for b = 1:opts.batchSize
                    [t2, c2] = tsp_twoopt_ls(problem, tours(b,:), opts.twoOptPasses);
                    tours(b,:) = t2;
                    costs(b)   = c2;
                end
            case 'elite'
                [~, ordAll] = sort(costs, 'ascend');
                topIdx = ordAll(1:twoOptEliteCount);
                for ii = 1:numel(topIdx)
                    b = topIdx(ii);
                    [t2, c2] = tsp_twoopt_ls(problem, tours(b,:), opts.twoOptPasses);
                    tours(b,:) = t2;
                    costs(b)   = c2;
                end
            case 'none'
                % nothing
            otherwise
                error('Unknown twoOptMode: %s', opts.twoOptMode);
        end

        % --------------------
        % Update incumbent best
        % --------------------
        [minCost, minIdx] = min(costs);
        if minCost < bestCost
            bestCost = minCost;
            bestTour = tours(minIdx,:);
            lastImproveIter = it;
        end

        if opts.twoOptOnBest
            [bestTour, bestCost] = tsp_twoopt_ls(problem, bestTour, opts.twoOptPasses);
        end

        % --------------------
        % Elite selection + weighted frequency
        % --------------------
        [~, ord] = sort(costs, 'ascend');
        eliteIdx = ord(1:eliteCount);

        w = elite_weights(costs(eliteIdx), opts.eliteWeighting, opts.weightGamma);

        freq = zeros(n,n);
        for e = 1:eliteCount
            t = tours(eliteIdx(e),:);
            we = w(e);

            for i = 1:n-1
                a = t(i); bb = t(i+1);
                freq(a,bb) = freq(a,bb) + we;
                freq(bb,a) = freq(bb,a) + we;
            end
            a = t(n); bb = t(1);
            freq(a,bb) = freq(a,bb) + we;
            freq(bb,a) = freq(bb,a) + we;
        end
        freq(1:n+1:end) = 0;

        eliteP = row_normalize(freq);

        % Elitist reinforcement of incumbent best
        if opts.bestReinf > 0
            bestFreq = zeros(n,n);
            t = bestTour;
            for i = 1:n-1
                a = t(i); bb = t(i+1);
                bestFreq(a,bb) = bestFreq(a,bb) + 1;
                bestFreq(bb,a) = bestFreq(bb,a) + 1;
            end
            a = t(n); bb = t(1);
            bestFreq(a,bb) = bestFreq(a,bb) + 1;
            bestFreq(bb,a) = bestFreq(bb,a) + 1;

            bestP = row_normalize(bestFreq);
            eliteP = (1-opts.bestReinf)*eliteP + opts.bestReinf*bestP;
        end

        % Mix base distribution (keeps exploration and prevents collapse)
        if baseMixEff > 0
            eliteP = (1-baseMixEff)*eliteP + baseMixEff*baseP;
        end

        % Apply bounds + normalize
        eliteP = apply_prob_bounds(eliteP, opts.minProb, opts.maxProb);
        eliteP(1:n+1:end) = 0;
        eliteP = row_normalize(eliteP);

        % Current P from theta
        Pcur = exp(theta);
        Pcur(1:n+1:end) = 0;
        Pcur = row_normalize(Pcur);

        % EMA update in probability space
        Pnew = (1-alphaEff)*Pcur + alphaEff*eliteP;
        Pnew = apply_prob_bounds(Pnew, opts.minProb, opts.maxProb);
        Pnew(1:n+1:end) = 0;
        Pnew = row_normalize(Pnew);

        theta = log(Pnew + 1e-12);
        theta(1:n+1:end) = -inf;

        curveBest(it) = bestCost;
    end

    info = struct();
    info.curveBest = curveBest;
    info.timeSec   = toc(t0);
    info.seed      = seed;
    info.opts      = opts;
end

% --------------------
% Sampling
% --------------------
function tour = sample_tour(theta, etaLog, candList, startNode, tau, beta)
    n = size(theta,1);

    tour = zeros(1,n);
    visited = false(1,n);

    cur = startNode;
    tour(1) = cur;
    visited(cur) = true;

    for k = 2:n
        unvis = find(~visited);

        if ~isempty(candList{cur})
            cand = candList{cur};
            cand = cand(~visited(cand));
            if ~isempty(cand)
                unvisUse = cand;
            else
                unvisUse = unvis;
            end
        else
            unvisUse = unvis;
        end

        scores = theta(cur, unvisUse) + beta * etaLog(cur, unvisUse);
        scores = scores / max(tau, 1e-12);
        scores = scores - max(scores); % stabilize

        p = exp(scores);
        p = p / sum(p);

        r = rand();
        cdf = cumsum(p);
        j = find(cdf >= r, 1, 'first');
        nxt = unvisUse(j);

        tour(k) = nxt;
        visited(nxt) = true;
        cur = nxt;
    end
end

% --------------------
% Schedules and utils
% --------------------
function tau = get_tau(opts, it)
    t0 = opts.tauStart;
    t1 = opts.tauEnd;
    K  = max(1, opts.maxIter);

    switch lower(opts.tauSchedule)
        case 'exp'
            c = 5 / max(1, K-1);
            tau = t1 + (t0 - t1) * exp(-c*(it-1));
        otherwise % 'linear'
            tau = t0 + (t1 - t0) * (it-1) / max(1, K-1);
    end
    tau = max(tau, 1e-6);
end

function baseP = build_baseP(D)
    % baseP(i,:) is a row-stochastic distribution preferring short edges from i
    n = size(D,1);
    s = mean(D(:) + eps);
    W = exp(-(D / max(s,eps)));
    W(1:n+1:end) = 0;
    baseP = row_normalize(W);
end

function P = row_normalize(W)
    n = size(W,1);
    P = W;
    P(1:n+1:end) = 0;
    rs = sum(P,2);
    rs(rs == 0) = 1;
    P = P ./ rs;
end

function P = apply_prob_bounds(P, minProb, maxProb)
    if nargin < 2 || isempty(minProb), minProb = 0; end
    if nargin < 3 || isempty(maxProb), maxProb = 1; end
    maxProb = min(maxProb, 1);

    n = size(P,1);
    mask = true(n,n);
    mask(1:n+1:end) = false;

    if minProb > 0
        P(mask) = max(P(mask), minProb);
    end
    if maxProb < 1
        P(mask) = min(P(mask), maxProb);
    end
end

function w = elite_weights(eliteCosts, mode, gamma)
    m = numel(eliteCosts);
    switch lower(mode)
        case 'uniform'
            w = ones(m,1) / m;
        case 'rank'
            r = (1:m).';
            w = 1 ./ r;
            w = w / sum(w);
        case 'exp'
            c0 = eliteCosts(1);
            denom = max(abs(c0), 1e-12);
            z = (eliteCosts - c0) / denom;
            w = exp(-gamma * z);
            w = w / sum(w);
        otherwise
            w = ones(m,1) / m;
    end
end

function s = set_defaults(s, d)
    if isempty(s), s = struct(); end
    f = fieldnames(d);
    for i = 1:numel(f)
        if ~isfield(s,f{i}) || isempty(s.(f{i}))
            s.(f{i}) = d.(f{i});
        end
    end
end
