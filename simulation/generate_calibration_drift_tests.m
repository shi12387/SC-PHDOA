function generate_calibration_drift_tests
% Physical context-target calibration-mismatch test sets.
%
% Context scenes use the base calibration state.  Guard and target scenes
% use a drifted state, allowing the frozen guard residual router to observe
% the same mismatch that affects target inference.  All drift directions
% are sampled independently of source, SNR, DOA, waveform, and scene role.

components = {'gain_phase', 'coupling', 'position', 'compound'};
for component_index = 1:numel(components)
    generate_one_component_local(components{component_index});
end
end


function generate_one_component_local(drift_component)
% It creates 300 new array states, balanced across case and drift strength.
% Each state contains the complete K={1,2,3} x SNR={-5,0,5,10,15,20}
% factorial.  The 8 context, 4 guard, and 6 target scenes are fixed in the
% MAT file.  Within every impairment case, the target assignment cycles so
% that every K/SNR cell appears exactly 20 times in the final target split.
%
% IMPORTANT: generate this set only after all model weights, thresholds,
% ALS settings, metrics, and baselines have been frozen.  Do not use its
% labels to retrain or select hyperparameters.
close all;

%% User-editable paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
default_data_dir = fullfile(fileparts(script_dir), 'data');

% Reusing a known protocol preamble is not sample leakage.  Only this fixed
% preamble is read from the training MAT; no training IQ or labels are used.
reference_train_mat = fullfile(default_data_dir, ...
    'session_calibration_training.mat');
output_dir = default_data_dir;

%% Locked calibration-mismatch test configuration
generator_seed = 18020;
rng(generator_seed, 'twister');

component_names = {'gain_phase', 'coupling', 'position', 'compound'};
component_id = find(strcmp(component_names, drift_component), 1);
if isempty(component_id)
    error('Unknown calibration-mismatch component: %s', drift_component);
end

% Base state is sampled from exactly the locked ID impairment support.
severity_bounds = [0.02, 0.12; 2.0, 12.0; ...
    0.03, 0.22; 0.005, 0.060];
drift_lambdas = [0, 0.05, 0.10, 0.20, 0.30];
full_gain_drift_sigma = 0.12;
full_phase_drift_sigma_deg = 12.0;
full_coupling_drift_amp = 0.22;
full_position_drift_sigma_d = 0.03;

num_array_states = 300;
state_case_counts = 60*ones(1, 5);
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
    error('Reference training MAT not found: %s', reference_train_mat);
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

% Twelve sessions per base-impairment/drift-strength stratum.  The role
% cycle resets per stratum to retain exact target K/SNR balance.
state_case_sequence = zeros(1, num_array_states);
state_lambda_index = zeros(1, num_array_states);
cursor = 0;
for case_id = 0:4
    for lambda_index = 1:numel(drift_lambdas)
        positions = cursor+(1:12);
        state_case_sequence(positions) = case_id;
        state_lambda_index(positions) = lambda_index;
        cursor = cursor+12;
    end
end
permutation = randperm(num_array_states);
state_case_sequence = state_case_sequence(permutation);
state_lambda_index = state_lambda_index(permutation);
stratum_occurrence = zeros(5, numel(drift_lambdas));

output_name = ['drift_', drift_component, '.mat'];
filename = fullfile(output_dir, output_name);
partial_filename = [filename, '.partial.mat'];
if exist(filename, 'file')
    error('Final output already exists: %s', filename);
end
if exist(partial_filename, 'file')
    error(['Partial output exists: %s\nDelete or rename it before ' ...
        'starting a clean deterministic generation.'], partial_filename);
end

fprintf('========== %s Calibration-Mismatch Generator ==========\n', ...
    drift_component);
fprintf('reference preamble = %s\n', reference_train_mat);
fprintf('output = %s\n', filename);
fprintf('states/case = %s | total states = %d\n', ...
    mat2str(state_case_counts), num_array_states);
fprintf('K = %s | SNR = %s | scenes/state = %d\n', ...
    mat2str(K_values), mat2str(snr_levels), scenes_per_state);
fprintf('drift lambdas = %s | context=base, guard/target=drifted\n', ...
    mat2str(drift_lambdas));
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
store.coupling_pre_norm_all(num_samples, 1) = single(0);
store.coupling_post_norm_all(num_samples, 1) = single(0);
store.coupling_condition_all(num_samples, 1) = single(0);
store.coupling_min_singular_all(num_samples, 1) = single(0);
store.coupling_was_scaled_all(num_samples, 1) = single(0);
store.max_position_offset_d_all(num_samples, 1) = single(0);
store.min_sensor_gain_amp_all(num_samples, 1) = single(0);
store.max_sensor_gain_amp_all(num_samples, 1) = single(0);
store.drift_lambda_all(num_samples, 1) = single(0);
store.drift_component_id_all(num_samples, 1) = single(0);
store.scene_calibration_stage_all(num_samples, 1) = single(0);
store.gain_phase_drift_norm_all(num_samples, 1) = single(0);
store.coupling_drift_norm_all(num_samples, 1) = single(0);
store.position_drift_rms_d_all(num_samples, 1) = single(0);
store.manifold_drift_all(num_samples, 1) = single(0);
store.context_sensor_gain_real(num_samples, num_ant) = single(0);
store.context_sensor_gain_imag(num_samples, num_ant) = single(0);
store.context_coupling_real(num_samples, num_ant, num_ant) = single(0);
store.context_coupling_imag(num_samples, num_ant, num_ant) = single(0);
store.context_array_coords_all(num_samples, 3, num_ant) = single(0);

sample_idx = 0;
generation_start = tic;
for state_index = 1:num_array_states
    case_id = state_case_sequence(state_index);
    case_slot = case_id+1;
    lambda_index = state_lambda_index(state_index);
    drift_lambda = drift_lambdas(lambda_index);
    stratum_occurrence(case_slot, lambda_index) = ...
        stratum_occurrence(case_slot, lambda_index)+1;
    occurrence = stratum_occurrence(case_slot, lambda_index);
    role_cycle = mod(occurrence-1, 3);
    roles = balanced_scene_roles_local( ...
        scenes_per_state, role_cycle, occurrence);

    [gain_sigma, phase_sigma, coupling_amp, position_sigma] = ...
        sample_state_severity_local(case_id, severity_bounds);
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
    [coupling, coupling_pre_norm, coupling_was_scaled] = ...
        coupling_matrix_local( ...
        actual_coords, d, coupling_amp, ...
        coupling_phase, coupling_scale);
    coupling_post_norm = norm(coupling-eye(num_ant), 2);
    coupling_singular = svd(coupling);
    coupling_condition = max(coupling_singular)/ ...
        max(min(coupling_singular), 1e-12);
    coupling_min_singular = min(coupling_singular);
    max_position_offset_d = max(vecnorm( ...
        actual_coords-nominal_array_coords, 2, 1))/d;
    sensor_gain_amplitude = abs(sensor_gain);

    % Draw all component directions for every dataset so lambda=0 sessions
    % remain distribution-matched across the four component files.
    drift_gain_draw = randn(1, num_ant);
    drift_phase_draw = randn(1, num_ant);
    drift_gain_draw(1) = 0;
    drift_phase_draw(1) = 0;
    drift_position_draw = randn(3, num_ant);
    drift_position_draw(2, :) = 0;
    drift_position_draw(:, 1) = 0;
    drift_coupling_phase = 2*pi*rand(num_ant, num_ant);
    drift_coupling_phase = triu(drift_coupling_phase, 1);
    drift_coupling_phase = drift_coupling_phase+drift_coupling_phase.';
    drift_coupling_scale = 0.70+0.60*rand(num_ant, num_ant);
    drift_coupling_scale = triu(drift_coupling_scale, 1);
    drift_coupling_scale = drift_coupling_scale+drift_coupling_scale.';
    [drift_template, ~, ~] = coupling_matrix_local( ...
        actual_coords, d, full_coupling_drift_amp, ...
        drift_coupling_phase, drift_coupling_scale);

    apply_gain = strcmp(drift_component, 'gain_phase') || ...
        strcmp(drift_component, 'compound');
    apply_coupling = strcmp(drift_component, 'coupling') || ...
        strcmp(drift_component, 'compound');
    apply_position = strcmp(drift_component, 'position') || ...
        strcmp(drift_component, 'compound');

    drifted_gain_amplitude = 1+gain_sigma*gain_draw;
    drifted_phase_deg = phase_sigma*phase_draw;
    if apply_gain
        drifted_gain_amplitude = drifted_gain_amplitude + ...
            drift_lambda*full_gain_drift_sigma*drift_gain_draw;
        drifted_phase_deg = drifted_phase_deg + ...
            drift_lambda*full_phase_drift_sigma_deg*drift_phase_draw;
    end
    drifted_sensor_gain = drifted_gain_amplitude.* ...
        exp(1j*deg2rad(drifted_phase_deg));

    drifted_coords = actual_coords;
    if apply_position
        drifted_coords = drifted_coords + drift_lambda* ...
            full_position_drift_sigma_d*d*drift_position_draw;
    end
    drifted_coupling = coupling;
    if apply_coupling
        drifted_coupling = coupling + ...
            drift_lambda*(drift_template-eye(num_ant));
        drifted_coupling = stabilize_coupling_local( ...
            drifted_coupling, 0.65);
    end

    gain_phase_drift_norm = norm( ...
        drifted_sensor_gain-sensor_gain)/max(norm(sensor_gain), 1e-12);
    coupling_drift_norm = norm( ...
        drifted_coupling-coupling, 'fro')/ ...
        max(norm(coupling, 'fro'), 1e-12);
    position_drift_rms_d = sqrt(mean(sum( ...
        (drifted_coords-actual_coords).^2, 1)))/d;
    manifold_drift = manifold_drift_local( ...
        sensor_gain, coupling, actual_coords, ...
        drifted_sensor_gain, drifted_coupling, drifted_coords, ...
        lambda_val);

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
    gain_real_block = zeros(scenes_per_state, num_ant, 'single');
    gain_imag_block = zeros(scenes_per_state, num_ant, 'single');
    coupling_real_block = zeros( ...
        scenes_per_state, num_ant, num_ant, 'single');
    coupling_imag_block = zeros( ...
        scenes_per_state, num_ant, num_ant, 'single');
    coords_block = zeros(scenes_per_state, 3, num_ant, 'single');

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

            if roles(scene_index) == 0
                scene_gain = sensor_gain;
                scene_coupling = coupling;
                scene_coords = actual_coords;
            else
                scene_gain = drifted_sensor_gain;
                scene_coupling = drifted_coupling;
                scene_coords = drifted_coords;
            end
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
                    scene_coords, lambda_val, ...
                    az(source), el(source));
            end
            effective_manifold = scene_coupling*(scene_gain(:).*A);
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
            gain_real_block(scene_index, :) = single(real(scene_gain));
            gain_imag_block(scene_index, :) = single(imag(scene_gain));
            coupling_real_block(scene_index, :, :) = reshape( ...
                single(real(scene_coupling)), [1, num_ant, num_ant]);
            coupling_imag_block(scene_index, :, :) = reshape( ...
                single(imag(scene_coupling)), [1, num_ant, num_ant]);
            coords_block(scene_index, :, :) = reshape( ...
                single(scene_coords), [1, 3, num_ant]);
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
    store.sensor_gain_real(rows, :) = gain_real_block;
    store.sensor_gain_imag(rows, :) = gain_imag_block;
    store.coupling_real(rows, :, :) = coupling_real_block;
    store.coupling_imag(rows, :, :) = coupling_imag_block;
    store.actual_array_coords_all(rows, :, :) = coords_block;
    store.coupling_pre_norm_all(rows, :) = single( ...
        coupling_pre_norm*ones(scenes_per_state, 1));
    store.coupling_post_norm_all(rows, :) = single( ...
        coupling_post_norm*ones(scenes_per_state, 1));
    store.coupling_condition_all(rows, :) = single( ...
        coupling_condition*ones(scenes_per_state, 1));
    store.coupling_min_singular_all(rows, :) = single( ...
        coupling_min_singular*ones(scenes_per_state, 1));
    store.coupling_was_scaled_all(rows, :) = single( ...
        coupling_was_scaled*ones(scenes_per_state, 1));
    store.max_position_offset_d_all(rows, :) = single( ...
        max_position_offset_d*ones(scenes_per_state, 1));
    store.min_sensor_gain_amp_all(rows, :) = single( ...
        min(sensor_gain_amplitude)*ones(scenes_per_state, 1));
    store.max_sensor_gain_amp_all(rows, :) = single( ...
        max(sensor_gain_amplitude)*ones(scenes_per_state, 1));
    store.drift_lambda_all(rows, :) = single( ...
        drift_lambda*ones(scenes_per_state, 1));
    store.drift_component_id_all(rows, :) = single( ...
        component_id*ones(scenes_per_state, 1));
    store.scene_calibration_stage_all(rows, :) = single(roles(:) ~= 0);
    store.gain_phase_drift_norm_all(rows, :) = single( ...
        gain_phase_drift_norm*ones(scenes_per_state, 1));
    store.coupling_drift_norm_all(rows, :) = single( ...
        coupling_drift_norm*ones(scenes_per_state, 1));
    store.position_drift_rms_d_all(rows, :) = single( ...
        position_drift_rms_d*ones(scenes_per_state, 1));
    store.manifold_drift_all(rows, :) = single( ...
        manifold_drift*ones(scenes_per_state, 1));
    store.context_sensor_gain_real(rows, :) = repmat( ...
        single(real(sensor_gain)), scenes_per_state, 1);
    store.context_sensor_gain_imag(rows, :) = repmat( ...
        single(imag(sensor_gain)), scenes_per_state, 1);
    store.context_coupling_real(rows, :, :) = repmat( ...
        reshape(single(real(coupling)), [1, num_ant, num_ant]), ...
        scenes_per_state, 1, 1);
    store.context_coupling_imag(rows, :, :) = repmat( ...
        reshape(single(imag(coupling)), [1, num_ant, num_ant]), ...
        scenes_per_state, 1, 1);
    store.context_array_coords_all(rows, :, :) = repmat( ...
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
store.state_lambda_index = state_lambda_index;
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
store.dataset_version = single(1830);
store.dataset_role = 'never_observed_external_paper_test';
store.b3_drift_component = drift_component;
store.b3_drift_component_id = component_id;
store.b3_drift_lambdas = drift_lambdas;
store.b3_base_impairment_bounds = severity_bounds;
store.b3_full_gain_drift_sigma = full_gain_drift_sigma;
store.b3_full_phase_drift_sigma_deg = full_phase_drift_sigma_deg;
store.b3_full_coupling_drift_amp = full_coupling_drift_amp;
store.b3_full_position_drift_sigma_d = full_position_drift_sigma_d;
store.coupling_spectral_norm_cap = 0.65;
store.reference_preamble_source = reference_train_mat;
clear store reference;
movefile(partial_filename, filename);

fprintf('\nTarget balance per case: 20 samples for every K/SNR cell.\n');
fprintf('Each drift lambda has 60 sessions and 360 target scenes.\n');
fprintf('Done: %s\n', filename);
fprintf('Total generation time: %.1f min\n', toc(generation_start)/60);
end


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
        sample_state_severity_local(case_id, bounds)
    gain_sigma = 0;
    phase_sigma = 0;
    coupling_amp = 0;
    position_sigma = 0;
    if case_id == 1 || case_id == 4
        gain_sigma = bounds(1, 1)+diff(bounds(1, :))*rand();
        phase_sigma = bounds(2, 1)+diff(bounds(2, :))*rand();
    end
    if case_id == 2 || case_id == 4
        coupling_amp = bounds(3, 1)+diff(bounds(3, :))*rand();
    end
    if case_id == 3 || case_id == 4
        position_sigma = bounds(4, 1)+diff(bounds(4, :))*rand();
    end
end


function [coupling, pre_norm, was_scaled] = coupling_matrix_local( ...
        coords, spacing, nearest_amp, phase_draw, amplitude_draw)
    count = size(coords, 2);
    coupling = eye(count);
    pre_norm = 0;
    was_scaled = false;
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
    pre_norm = norm(coupling-eye(count), 2);
    if pre_norm > 0.65
        coupling = eye(count)+(coupling-eye(count))* ...
            (0.65/pre_norm);
        was_scaled = true;
    end
end


function coupling = stabilize_coupling_local(coupling, cap)
    count = size(coupling, 1);
    residual = coupling-eye(count);
    spectral_norm = norm(residual, 2);
    if spectral_norm > cap
        coupling = eye(count)+residual*(cap/spectral_norm);
    end
end


function value = manifold_drift_local( ...
        context_gain, context_coupling, context_coords, ...
        target_gain, target_coupling, target_coords, wavelength)
    az_grid = [-45, 0, 45];
    el_grid = [15, 45, 75];
    context = complex(zeros(numel(az_grid)*numel(el_grid), ...
        numel(context_gain)));
    target = context;
    cursor = 0;
    for az = az_grid
        for el = el_grid
            cursor = cursor+1;
            a_context = steering_local( ...
                context_coords, wavelength, az, el);
            a_target = steering_local( ...
                target_coords, wavelength, az, el);
            context(cursor, :) = (context_coupling*( ...
                context_gain(:).*a_context)).';
            target(cursor, :) = (target_coupling*( ...
                target_gain(:).*a_target)).';
        end
    end
    value = norm(target-context, 'fro')/ ...
        max(norm(context, 'fro'), 1e-12);
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
