% Final never-observed external paper test set.
%
% This generator is intentionally separate from every training generator.
% It creates 300 new array states, balanced across five impairment cases.
% Each state contains the complete K={1,2,3} x SNR={-5,0,5,10,15,20}
% factorial.  The 8 context, 4 guard, and 6 target scenes are fixed in the
% MAT file.  Within every impairment case, the target assignment cycles so
% that every K/SNR cell appears exactly 20 times in the final target split.
%
% IMPORTANT: generate this set only after all model weights, thresholds,
% ALS settings, metrics, and baselines have been frozen.  Do not use its
% labels to retrain or select hyperparameters.
clc; clear; close all;

%% User-editable paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
default_data_dir = fullfile(fileparts(script_dir), 'data');

% Reusing a known protocol preamble is not sample leakage.  Only this fixed
% preamble is read from the training MAT; no training IQ or labels are used.
reference_train_mat = fullfile(default_data_dir, ...
    'session_calibration_training.mat');
output_dir = default_data_dir;

%% Locked final-test configuration
generator_seed = 18020;
rng(generator_seed, 'twister');

num_array_states = 300;
state_case_counts = 60*ones(1, 5);
state_case_names = { ...
    'ideal', 'gain_phase', 'coupling', 'position', 'compound'};
K_values = [1, 2, 3];
snr_levels = [-5, 0, 5, 10, 15, 20];
scenes_per_state = numel(K_values)*numel(snr_levels);  % 18
num_samples = num_array_states*scenes_per_state;
L = 1024;
preamble_len = 256;
max_timing_offset = 128;
cfo_min = -0.01;
cfo_max = 0.01;

if sum(state_case_counts) ~= num_array_states
    error('state_case_counts must sum to num_array_states.');
end
if any(mod(state_case_counts, 3) ~= 0)
    error('Each case count must be divisible by 3 for target balancing.');
end
if ~exist(reference_train_mat, 'file')
    error(['Reference training MAT not found:\n%s\n' ...
        'Edit reference_train_mat at the top of this script.'], ...
        reference_train_mat);
end
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

reference = matfile(reference_train_mat);
preamble_real = reference.shared_preamble_real;
preamble_imag = reference.shared_preamble_imag;
shared_preamble = double(preamble_real(:).') + ...
    1j*double(preamble_imag(:).');
if numel(shared_preamble) ~= preamble_len
    error('Expected a %d-sample shared preamble, found %d.', ...
        preamble_len, numel(shared_preamble));
end
shared_preamble = shared_preamble ./ ...
    sqrt(mean(abs(shared_preamble).^2)+1e-12);

%% Array and waveform configuration locked to the training support
c = 3e8;
f_c = 5.8e9;
lambda_val = c/f_c;
d = lambda_val/2;
num_ant = 7;
max_sources = 3;
nominal_array_coords = [ ...
    0, d, 2*d, 3*d, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, d, 2*d, 3*d];

family_probs = [0.45, 0.25, 0.30];
sps_values = [4, 8];
rrc_rolloff_range = [0.20, 0.50];
rrc_span = 8;
cpfsk_h_range = [0.35, 0.70];
ofdm_fft_values = [64, 128];
ofdm_cp_values = [1/8, 1/4];

% Exactly balanced case membership, then randomly interleaved.  Role-cycle
% counters are maintained per case, not by global state index.
state_case_sequence = [ ...
    zeros(1, state_case_counts(1)), ...
    ones(1, state_case_counts(2)), ...
    2*ones(1, state_case_counts(3)), ...
    3*ones(1, state_case_counts(4)), ...
    4*ones(1, state_case_counts(5))];
state_case_sequence = state_case_sequence(randperm(num_array_states));
case_occurrence = zeros(1, 5);

filename = fullfile(output_dir, ...
    'external_session_test_L1024_states300_scenes18_N5400_P256.mat');
partial_filename = [filename, '.partial.mat'];
if exist(filename, 'file')
    error('Final output already exists: %s', filename);
end
if exist(partial_filename, 'file')
    error(['Partial output exists: %s\nDelete or rename it before ' ...
        'starting a clean deterministic generation.'], partial_filename);
end

fprintf('========== Final External Paper Test Generator ==========\n');
fprintf('reference preamble = %s\n', reference_train_mat);
fprintf('output = %s\n', filename);
fprintf('states/case = %s | total states = %d\n', ...
    mat2str(state_case_counts), num_array_states);
fprintf('K = %s | SNR = %s | scenes/state = %d\n', ...
    mat2str(K_values), mat2str(snr_levels), scenes_per_state);
fprintf('N = %d | L = %d | P = %d | seed = %d\n', ...
    num_samples, L, preamble_len, generator_seed);

%% HDF5-backed, resumable-safe storage
store = matfile(partial_filename, 'Writable', true);
store.IQ_tensors(num_samples, L, num_ant, 2) = single(0);
store.source_waveforms_real(num_samples, max_sources, L) = single(0);
store.source_waveforms_imag(num_samples, max_sources, L) = single(0);
store.doas_all(num_samples, max_sources, 2) = single(nan);
store.timing_offset_all(num_samples, max_sources) = single(nan);
store.cfo_all(num_samples, max_sources) = single(nan);
store.source_power_db_all(num_samples, max_sources) = single(nan);
store.snr_all(num_samples, 1) = single(0);
store.array_state_id_all(num_samples, 1) = single(0);
store.scene_within_state_all(num_samples, 1) = single(0);
store.geometry_id_all(num_samples, 1) = single(0);
store.num_sources_all(num_samples, 1) = single(0);
store.difficulty_all(num_samples, 1) = single(0);
store.min_sep_all(num_samples, 1) = single(nan);
store.impairment_case_all(num_samples, 1) = single(0);
store.evaluation_role_all(num_samples, 1) = single(0);
store.role_cycle_all(num_samples, 1) = single(0);
store.gain_sigma_all(num_samples, 1) = single(0);
store.phase_sigma_deg_all(num_samples, 1) = single(0);
store.coupling_amp_all(num_samples, 1) = single(0);
store.position_sigma_d_all(num_samples, 1) = single(0);
store.sensor_gain_real(num_samples, num_ant) = single(0);
store.sensor_gain_imag(num_samples, num_ant) = single(0);
store.coupling_real(num_samples, num_ant, num_ant) = single(0);
store.coupling_imag(num_samples, num_ant, num_ant) = single(0);
store.actual_array_coords_all(num_samples, 3, num_ant) = single(0);

sample_idx = 0;
generation_start = tic;
for state_index = 1:num_array_states
    case_id = state_case_sequence(state_index);
    case_slot = case_id+1;
    case_occurrence(case_slot) = case_occurrence(case_slot)+1;
    occurrence = case_occurrence(case_slot);
    role_cycle = mod(occurrence-1, 3);
    roles = balanced_scene_roles_local( ...
        scenes_per_state, role_cycle, occurrence);

    [gain_sigma, phase_sigma, coupling_amp, position_sigma] = ...
        sample_state_severity_local(case_id);
    gain_draw = randn(1, num_ant);
    phase_draw = randn(1, num_ant);
    gain_draw(1) = 0;
    phase_draw(1) = 0;
    sensor_gain = (1+gain_sigma*gain_draw).* ...
        exp(1j*deg2rad(phase_sigma*phase_draw));

    position_draw = randn(3, num_ant);
    position_draw(2, :) = 0;
    position_draw(:, 1) = 0;
    actual_coords = nominal_array_coords + ...
        position_sigma*d*position_draw;

    coupling_phase = 2*pi*rand(num_ant, num_ant);
    coupling_phase = triu(coupling_phase, 1);
    coupling_phase = coupling_phase+coupling_phase.';
    coupling_scale = 0.70+0.60*rand(num_ant, num_ant);
    coupling_scale = triu(coupling_scale, 1);
    coupling_scale = coupling_scale+coupling_scale.';
    coupling = coupling_matrix_local( ...
        actual_coords, d, coupling_amp, ...
        coupling_phase, coupling_scale);

    rows = sample_idx+(1:scenes_per_state);
    iq_block = zeros(scenes_per_state, L, num_ant, 2, 'single');
    source_real_block = zeros( ...
        scenes_per_state, max_sources, L, 'single');
    source_imag_block = zeros( ...
        scenes_per_state, max_sources, L, 'single');
    doa_block = nan(scenes_per_state, max_sources, 2, 'single');
    timing_block = nan(scenes_per_state, max_sources, 'single');
    cfo_block = nan(scenes_per_state, max_sources, 'single');
    power_block = nan(scenes_per_state, max_sources, 'single');
    snr_block = zeros(scenes_per_state, 1, 'single');
    k_block = zeros(scenes_per_state, 1, 'single');
    difficulty_block = zeros(scenes_per_state, 1, 'single');
    min_sep_block = nan(scenes_per_state, 1, 'single');

    scene_index = 0;
    for K = K_values
        for snr_db = snr_levels
            scene_index = scene_index+1;
            difficulty = sample_discrete_local( ...
                [1, 2, 3], [0.20, 0.40, 0.40]);
            if difficulty == 1
                sep_req = 18; ux_req = 0.10; uz_req = 0.08;
            elseif difficulty == 2
                sep_req = 12; ux_req = 0.055; uz_req = 0.040;
            else
                sep_req = 7; ux_req = 0.025; uz_req = 0.020;
            end
            [az, el] = sample_geometry_local( ...
                K, sep_req, ux_req, uz_req);
            if K > 1
                min_sep = minimum_spherical_separation(az, el);
            else
                min_sep = nan;
            end

            [timing_offsets, cfo_values] = sample_sync_parameters( ...
                K, max_timing_offset, cfo_min, cfo_max);
            S = complex(zeros(K, L));
            for source = 1:K
                timing = timing_offsets(source);
                available = L-timing;
                payload = generate_structured_waveform( ...
                    available, family_probs, sps_values, ...
                    rrc_rolloff_range, rrc_span, cpfsk_h_range, ...
                    ofdm_fft_values, ofdm_cp_values, ...
                    [0, 0], [1, 1]);
                packet = payload;
                packet(1:preamble_len) = shared_preamble;
                n = 0:(available-1);
                phase0 = 2*pi*rand();
                packet = packet.*exp(1j*( ...
                    2*pi*cfo_values(source)*n+phase0));
                packet = packet/sqrt(mean(abs(packet).^2)+1e-12);
                S(source, timing+(1:available)) = packet;
            end

            A = complex(zeros(num_ant, K));
            for source = 1:K
                A(:, source) = steering_local( ...
                    actual_coords, lambda_val, ...
                    az(source), el(source));
            end
            effective_manifold = coupling*(sensor_gain(:).*A);
            Y_clean = effective_manifold*S;
            signal_power = mean(sum(abs(Y_clean).^2, 1));
            noise = (randn(num_ant, L)+1j*randn(num_ant, L))/sqrt(2);
            noise = noise/sqrt( ...
                mean(sum(abs(noise).^2, 1))+1e-12);
            noise = noise*sqrt(signal_power/(10^(snr_db/10)));
            Y = Y_clean+noise;

            iq_block(scene_index, :, :, 1) = ...
                reshape(single(real(Y).'), [1, L, num_ant, 1]);
            iq_block(scene_index, :, :, 2) = ...
                reshape(single(imag(Y).'), [1, L, num_ant, 1]);
            padded_S = complex(zeros(max_sources, L));
            padded_S(1:K, :) = S;
            source_real_block(scene_index, :, :) = ...
                reshape(single(real(padded_S)), [1, max_sources, L]);
            source_imag_block(scene_index, :, :) = ...
                reshape(single(imag(padded_S)), [1, max_sources, L]);
            doa_block(scene_index, 1:K, 1) = single(az);
            doa_block(scene_index, 1:K, 2) = single(el);
            timing_block(scene_index, 1:K) = single(timing_offsets);
            cfo_block(scene_index, 1:K) = single(cfo_values);
            power_block(scene_index, 1:K) = 0;
            snr_block(scene_index) = single(snr_db);
            k_block(scene_index) = single(K);
            difficulty_block(scene_index) = single(difficulty);
            min_sep_block(scene_index) = single(min_sep);
        end
    end

    store.IQ_tensors(rows, :, :, :) = iq_block;
    store.source_waveforms_real(rows, :, :) = source_real_block;
    store.source_waveforms_imag(rows, :, :) = source_imag_block;
    store.doas_all(rows, :, :) = doa_block;
    store.timing_offset_all(rows, :) = timing_block;
    store.cfo_all(rows, :) = cfo_block;
    store.source_power_db_all(rows, :) = power_block;
    store.snr_all(rows, :) = snr_block;
    store.array_state_id_all(rows, :) = ...
        single(state_index*ones(scenes_per_state, 1));
    store.scene_within_state_all(rows, :) = single((1:scenes_per_state).');
    store.geometry_id_all(rows, :) = single(rows.');
    store.num_sources_all(rows, :) = k_block;
    store.difficulty_all(rows, :) = difficulty_block;
    store.min_sep_all(rows, :) = min_sep_block;
    store.impairment_case_all(rows, :) = ...
        single(case_id*ones(scenes_per_state, 1));
    store.evaluation_role_all(rows, :) = single(roles(:));
    store.role_cycle_all(rows, :) = ...
        single(role_cycle*ones(scenes_per_state, 1));
    store.gain_sigma_all(rows, :) = ...
        single(gain_sigma*ones(scenes_per_state, 1));
    store.phase_sigma_deg_all(rows, :) = ...
        single(phase_sigma*ones(scenes_per_state, 1));
    store.coupling_amp_all(rows, :) = ...
        single(coupling_amp*ones(scenes_per_state, 1));
    store.position_sigma_d_all(rows, :) = ...
        single(position_sigma*ones(scenes_per_state, 1));
    store.sensor_gain_real(rows, :) = repmat( ...
        single(real(sensor_gain)), scenes_per_state, 1);
    store.sensor_gain_imag(rows, :) = repmat( ...
        single(imag(sensor_gain)), scenes_per_state, 1);
    store.coupling_real(rows, :, :) = repmat( ...
        reshape(single(real(coupling)), [1, num_ant, num_ant]), ...
        scenes_per_state, 1, 1);
    store.coupling_imag(rows, :, :) = repmat( ...
        reshape(single(imag(coupling)), [1, num_ant, num_ant]), ...
        scenes_per_state, 1, 1);
    store.actual_array_coords_all(rows, :, :) = repmat( ...
        reshape(single(actual_coords), [1, 3, num_ant]), ...
        scenes_per_state, 1, 1);
    sample_idx = sample_idx+scenes_per_state;

    if mod(state_index, 5) == 0 || state_index == num_array_states
        elapsed = toc(generation_start);
        rate = state_index/max(elapsed, 1e-9);
        remaining = (num_array_states-state_index)/max(rate, 1e-9);
        fprintf(['  state %d/%d | samples %d/%d | ' ...
            'elapsed %.1f min | ETA %.1f min\n'], ...
            state_index, num_array_states, sample_idx, num_samples, ...
            elapsed/60, remaining/60);
    end
end

%% Metadata and atomic finalize
store.shared_preamble_real = single(real(shared_preamble));
store.shared_preamble_imag = single(imag(shared_preamble));
store.nominal_array_coords = nominal_array_coords;
store.state_case_sequence = state_case_sequence;
store.state_case_names_char = char( ...
    'ideal     ', 'gain_phase', 'coupling  ', ...
    'position  ', 'compound  ');
store.evaluation_role_names_char = char( ...
    'context', 'guard  ', 'target ');
store.f_c = f_c;
store.c = c;
store.lambda_val = lambda_val;
store.d = d;
store.L = L;
store.max_sources = max_sources;
store.K_values = K_values;
store.snr_levels = snr_levels;
store.preamble_len = preamble_len;
store.max_timing_offset = max_timing_offset;
store.num_array_states = num_array_states;
store.scenes_per_state = scenes_per_state;
store.num_samples = num_samples;
store.generator_seed = generator_seed;
store.dataset_version = single(1802);
store.dataset_role = 'never_observed_external_paper_test';
store.reference_preamble_source = reference_train_mat;
clear store reference;
movefile(partial_filename, filename);

fprintf('\nTarget balance per case: 20 samples for every K/SNR cell.\n');
fprintf('Done: %s\n', filename);
fprintf('Total generation time: %.1f min\n', toc(generation_start)/60);


function roles = balanced_scene_roles_local(scene_count, cycle, occurrence)
    % role: 0=context, 1=guard, 2=target
    scene_zero = 0:(scene_count-1);
    target = find(mod(scene_zero, 3) == cycle);
    remaining = setdiff(1:scene_count, target, 'stable');
    guard_cycle = mod(floor((occurrence-1)/3), 3);
    guard = remaining((1:4)+4*guard_cycle);
    roles = zeros(scene_count, 1);
    roles(guard) = 1;
    roles(target) = 2;
    if sum(roles == 0) ~= 8 || sum(roles == 1) ~= 4 || ...
            sum(roles == 2) ~= 6
        error('Invalid context/guard/target role assignment.');
    end
end


function [gain_sigma, phase_sigma, coupling_amp, position_sigma] = ...
        sample_state_severity_local(case_id)
    gain_sigma = 0;
    phase_sigma = 0;
    coupling_amp = 0;
    position_sigma = 0;
    if case_id == 1 || case_id == 4
        gain_sigma = 0.02+0.10*rand();
        phase_sigma = 2.0+10.0*rand();
    end
    if case_id == 2 || case_id == 4
        coupling_amp = 0.03+0.19*rand();
    end
    if case_id == 3 || case_id == 4
        position_sigma = 0.005+0.055*rand();
    end
end


function coupling = coupling_matrix_local( ...
        coords, spacing, nearest_amp, phase_draw, amplitude_draw)
    count = size(coords, 2);
    coupling = eye(count);
    if nearest_amp <= 0; return; end
    for first = 1:count
        for second = first+1:count
            distance_units = norm( ...
                coords(:, first)-coords(:, second))/spacing;
            if distance_units <= 2.25
                amplitude = nearest_amp* ...
                    amplitude_draw(first, second)* ...
                    exp(-1.15*max(distance_units-1, 0));
                value = amplitude*exp(1j*phase_draw(first, second));
                coupling(first, second) = value;
                coupling(second, first) = value;
            end
        end
    end
    spectral_norm = norm(coupling-eye(count), 2);
    if spectral_norm > 0.65
        coupling = eye(count)+(coupling-eye(count))* ...
            (0.65/spectral_norm);
    end
end


function [timing, cfo] = sample_sync_parameters(K, max_delay, fmin, fmax)
    if K == 1
        timing = randi([0, max_delay], 1, 1);
        cfo = fmin+(fmax-fmin)*rand(1, 1);
        return;
    end
    for attempt = 1:10000
        timing = randi([0, max_delay], 1, K);
        cfo = fmin+(fmax-fmin)*rand(1, K);
        valid = true;
        for first = 1:K
            for second = first+1:K
                if abs(timing(first)-timing(second)) < 12 && ...
                        abs(cfo(first)-cfo(second)) < 0.0015
                    valid = false;
                end
            end
        end
        if valid; return; end
    end
    error('Unable to sample separable timing/CFO parameters.');
end


function value = sample_discrete_local(values, probabilities)
    cdf = cumsum(probabilities/sum(probabilities));
    value = values(find(rand() <= cdf, 1, 'first'));
end


function [az, el] = sample_geometry_local(K, sep_req, ux_req, uz_req)
    if K == 1
        az = -60+120*rand(1, 1);
        el = 5+75*rand(1, 1);
        return;
    end
    for attempt = 1:20000
        az = -60+120*rand(1, K);
        el = 5+75*rand(1, K);
        ux = cosd(el).*sind(az);
        uz = sind(el);
        if minimum_spherical_separation(az, el) < sep_req; continue; end
        if minimum_abs_difference(ux) < ux_req; continue; end
        if minimum_abs_difference(uz) < uz_req; continue; end
        return;
    end
    error('Unable to sample geometry for K=%d.', K);
end


function value = minimum_abs_difference(x)
    value = inf;
    for first = 1:numel(x)
        for second = first+1:numel(x)
            value = min(value, abs(x(first)-x(second)));
        end
    end
end


function value = minimum_spherical_separation(az, el)
    value = inf;
    for first = 1:numel(az)
        for second = first+1:numel(az)
            first_vector = [ ...
                cosd(el(first))*sind(az(first)); ...
                cosd(el(first))*cosd(az(first)); ...
                sind(el(first))];
            second_vector = [ ...
                cosd(el(second))*sind(az(second)); ...
                cosd(el(second))*cosd(az(second)); ...
                sind(el(second))];
            value = min(value, acosd(max(-1, min(1, ...
                real(first_vector'*second_vector)))));
        end
    end
end


function a = steering_local(coords, wavelength, az, el)
    ux = cosd(el)*sind(az);
    uz = sind(el);
    phase = coords(1, :)*ux+coords(3, :)*uz;
    a = exp(-1j*2*pi*phase/wavelength).';
end
