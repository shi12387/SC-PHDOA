% Training set for session-shared compound array calibration.
%
% Each array state has one fixed physical calibration and 18 scenes that
% exactly cover K={1,2,3} x SNR={-5,0,5,10,15,20}. DOAs, source waveforms,
% synchronization parameters, and noise change from scene to scene.
%
% This is the locked pilot distribution for testing whether unlabeled
% multi-scene context can identify array calibration and improve target-scene
% DOA estimation. Increase num_array_states only after this pilot succeeds.
clc; clear; close all;
rng(17010, 'twister');

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

%% Locked pilot configuration
output_dir = fullfile(fileparts(script_dir), 'data');
num_array_states = 1000;
snr_levels = [-5, 0, 5, 10, 15, 20];
K_values = [1, 2, 3];
scenes_per_state = numel(K_values)*numel(snr_levels);  % 18
num_samples = num_array_states*scenes_per_state;
L = 1024;
preamble_len = 256;
max_timing_offset = 128;
cfo_min = -0.01;
cfo_max = 0.01;

% State categories: ideal 10%, gain/phase 15%, coupling 15%,
% position 15%, and compound 45%.
state_case_counts = floor(num_array_states*[0.10, 0.15, 0.15, 0.15, 0.45]);
state_case_counts(end) = state_case_counts(end)+ ...
    num_array_states-sum(state_case_counts);
state_case_sequence = [ ...
    zeros(1, state_case_counts(1)), ...
    ones(1, state_case_counts(2)), ...
    2*ones(1, state_case_counts(3)), ...
    3*ones(1, state_case_counts(4)), ...
    4*ones(1, state_case_counts(5))];
state_case_sequence = state_case_sequence(randperm(num_array_states));
state_case_names = { ...
    'ideal', 'gain_phase', 'coupling', 'position', 'compound'};

%% Array configuration
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

%% Signal configuration
preamble_symbols = randi([0, 3], 1, preamble_len);
shared_preamble = exp(1j*(pi/4+(pi/2)*preamble_symbols));
shared_preamble_real = single(real(shared_preamble));
shared_preamble_imag = single(imag(shared_preamble));
family_probs = [0.45, 0.25, 0.30];
sps_values = [4, 8];
rrc_rolloff_range = [0.20, 0.50];
rrc_span = 8;
cpfsk_h_range = [0.35, 0.70];
ofdm_fft_values = [64, 128];
ofdm_cp_values = [1/8, 1/4];

if ~exist(output_dir, 'dir'); mkdir(output_dir); end
filename = fullfile(output_dir, 'session_calibration_training.mat');
partial_filename = [filename, '.partial.mat'];
if exist(filename, 'file')
    error('Output already exists: %s', filename);
end
if exist(partial_filename, 'file')
    error(['Partial output exists: %s\nDelete or rename it before ' ...
        'starting a clean deterministic generation.'], partial_filename);
end

fprintf(['Generating session-shared calibration training set\n' ...
    '  states=%d, scenes/state=%d, N=%d, L=%d\n' ...
    '  K=%s, SNR=%s\n'], num_array_states, scenes_per_state, ...
    num_samples, L, mat2str(K_values), mat2str(snr_levels));

%% HDF5-backed storage
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
store.geometry_within_k_all(num_samples, 1) = single(0);
store.num_sources_all(num_samples, 1) = single(0);
store.difficulty_all(num_samples, 1) = single(0);
store.min_sep_all(num_samples, 1) = single(nan);
store.impairment_case_all(num_samples, 1) = single(0);
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
    [gain_sigma, phase_sigma, coupling_amp, position_sigma] = ...
        sample_state_severity_local(case_id);

    % Draw one fixed array calibration for the entire state.
    gain_draw = randn(1, num_ant);
    phase_draw = randn(1, num_ant);
    gain_draw(1) = 0;
    phase_draw(1) = 0;
    sensor_gain = (1+gain_sigma*gain_draw).* ...
        exp(1j*deg2rad(phase_sigma*phase_draw));

    position_draw = randn(3, num_ant);
    position_draw(2, :) = 0;
    position_draw(:, 1) = 0;
    actual_coords = nominal_array_coords+ ...
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
            noise = (randn(num_ant, L)+ ...
                1j*randn(num_ant, L))/sqrt(2);
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
                reshape(single(real(padded_S)), ...
                [1, max_sources, L]);
            source_imag_block(scene_index, :, :) = ...
                reshape(single(imag(padded_S)), ...
                [1, max_sources, L]);
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

    repeated_gain_real = repmat( ...
        single(real(sensor_gain)), scenes_per_state, 1);
    repeated_gain_imag = repmat( ...
        single(imag(sensor_gain)), scenes_per_state, 1);
    repeated_coupling_real = repmat( ...
        reshape(single(real(coupling)), [1, num_ant, num_ant]), ...
        scenes_per_state, 1, 1);
    repeated_coupling_imag = repmat( ...
        reshape(single(imag(coupling)), [1, num_ant, num_ant]), ...
        scenes_per_state, 1, 1);
    repeated_coords = repmat( ...
        reshape(single(actual_coords), [1, 3, num_ant]), ...
        scenes_per_state, 1, 1);

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
    store.scene_within_state_all(rows, :) = ...
        single((1:scenes_per_state).');
    store.geometry_id_all(rows, :) = single(rows.');
    store.geometry_within_k_all(rows, :) = ...
        single((1:scenes_per_state).');
    store.num_sources_all(rows, :) = k_block;
    store.difficulty_all(rows, :) = difficulty_block;
    store.min_sep_all(rows, :) = min_sep_block;
    store.impairment_case_all(rows, :) = ...
        single(case_id*ones(scenes_per_state, 1));
    store.gain_sigma_all(rows, :) = ...
        single(gain_sigma*ones(scenes_per_state, 1));
    store.phase_sigma_deg_all(rows, :) = ...
        single(phase_sigma*ones(scenes_per_state, 1));
    store.coupling_amp_all(rows, :) = ...
        single(coupling_amp*ones(scenes_per_state, 1));
    store.position_sigma_d_all(rows, :) = ...
        single(position_sigma*ones(scenes_per_state, 1));
    store.sensor_gain_real(rows, :) = repeated_gain_real;
    store.sensor_gain_imag(rows, :) = repeated_gain_imag;
    store.coupling_real(rows, :, :) = repeated_coupling_real;
    store.coupling_imag(rows, :, :) = repeated_coupling_imag;
    store.actual_array_coords_all(rows, :, :) = repeated_coords;
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
store.shared_preamble_real = shared_preamble_real;
store.shared_preamble_imag = shared_preamble_imag;
store.nominal_array_coords = nominal_array_coords;
store.state_case_sequence = state_case_sequence;
store.state_case_names_char = char( ...
    'ideal     ', 'gain_phase', 'coupling  ', ...
    'position  ', 'compound  ');
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
store.generator_seed = 17010;
store.dataset_version = single(1701);
store.dataset_role = 'session_shared_calibration_training';
clear store;
movefile(partial_filename, filename);
fprintf('Done: %s\n', filename);
fprintf('Total generation time: %.1f min\n', toc(generation_start)/60);


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
                value = amplitude*exp( ...
                    1j*phase_draw(first, second));
                coupling(first, second) = value;
                coupling(second, first) = value;
            end
        end
    end
    spectral_norm = norm(coupling-eye(count), 2);
    if spectral_norm > 0.65
        coupling = eye(count)+ ...
            (coupling-eye(count))*(0.65/spectral_norm);
    end
end


function [timing, cfo] = sample_sync_parameters( ...
        K, max_delay, fmin, fmax)
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
