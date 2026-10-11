% Diverse equal-power K1/K2/K3 source-count training set.
%
% Design goal:
%   Increase independent DOA geometries and independent array-error
%   realizations, rather than adding repeated noise to the same geometry.
%
% Default scale:
%   600 geometries/K x 3 K values x 9 samples/geometry = 16200 samples.
%   Across 600 geometries, every case has exactly 100 samples at each
%   SNR in [-5, 0, 5, 10, 15, 20] dB.
%
% The output schema is consumed by the released source-count trainer.
% Data are written geometry-by-geometry through matfile to limit RAM usage.
clc; clear; close all;
rng(16100, 'twister');

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

%% User configuration
output_dir = fullfile(fileparts(script_dir), 'data');
num_geometries_per_k = 600;
snr_levels = [-5, 0, 5, 10, 15, 20];
L = 1024;
preamble_len = 256;
max_timing_offset = 128;
cfo_min = -0.01;
cfo_max = 0.01;

% Keep one realization per case in a geometry. The cache builder uses
% one oracle manifold per geometry/case; combined cases are emphasized later
% by the training sampler rather than duplicated here.
case_ids = 0:8;
case_repeats = ones(1, numel(case_ids));
case_sequence = expand_case_sequence_local(case_ids, case_repeats);
samples_per_geometry = numel(case_sequence);

% Continuous severity ranges. Each sample draws a new level inside its range.
% Columns: [gain sigma, phase sigma deg, coupling amplitude, position sigma/d].
case_minimum = [ ...
    0.00,  0.0, 0.00, 0.000; ... % 0 ideal
    0.02,  2.0, 0.00, 0.000; ... % 1 gain/phase mild
    0.06,  6.0, 0.00, 0.000; ... % 2 gain/phase severe
    0.00,  0.0, 0.03, 0.000; ... % 3 coupling mild
    0.00,  0.0, 0.10, 0.000; ... % 4 coupling severe
    0.00,  0.0, 0.00, 0.005; ... % 5 position mild
    0.00,  0.0, 0.00, 0.025; ... % 6 position severe
    0.02,  2.0, 0.03, 0.005; ... % 7 combined mild
    0.06,  6.0, 0.10, 0.025];    % 8 combined severe
case_maximum = [ ...
    0.00,  0.0, 0.00, 0.000; ...
    0.06,  6.0, 0.00, 0.000; ...
    0.12, 12.0, 0.00, 0.000; ...
    0.00,  0.0, 0.10, 0.000; ...
    0.00,  0.0, 0.22, 0.000; ...
    0.00,  0.0, 0.00, 0.025; ...
    0.00,  0.0, 0.00, 0.060; ...
    0.06,  6.0, 0.10, 0.025; ...
    0.12, 12.0, 0.22, 0.060];

%% Array configuration
c = 3e8;
f_c = 5.8e9;
lambda_val = c/f_c;
d = lambda_val/2;
num_ant = 7;
max_sources = 3;
K_values = [1, 2, 3];
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

total_geometries = num_geometries_per_k*numel(K_values);
num_samples = total_geometries*samples_per_geometry;
if ~exist(output_dir, 'dir'); mkdir(output_dir); end
filename = fullfile(output_dir, 'source_count_training.mat');
partial_filename = [filename, '.partial.mat'];
if exist(filename, 'file')
    error('Output already exists: %s', filename);
end
if exist(partial_filename, 'file')
    error(['Partial output exists: %s\nDelete or rename it before ' ...
        'starting a clean deterministic generation.'], partial_filename);
end

fprintf(['Generating diverse source-count training set\n' ...
    '  K=%s, geometries/K=%d, total geometries=%d\n' ...
    '  samples/geometry=%d, total samples=%d, L=%d\n' ...
    '  cases=%s, repeats=%s, SNR=%s\n'], ...
    mat2str(K_values), num_geometries_per_k, total_geometries, ...
    samples_per_geometry, num_samples, L, mat2str(case_ids), ...
    mat2str(case_repeats), mat2str(snr_levels));

%% Preallocate an HDF5-backed MAT file without holding the dataset in RAM
store = matfile(partial_filename, 'Writable', true);
store.IQ_tensors(num_samples, L, num_ant, 2) = single(0);
store.source_waveforms_real(num_samples, max_sources, L) = single(0);
store.source_waveforms_imag(num_samples, max_sources, L) = single(0);
store.doas_all(num_samples, max_sources, 2) = single(nan);
store.timing_offset_all(num_samples, max_sources) = single(nan);
store.cfo_all(num_samples, max_sources) = single(nan);
store.source_power_db_all(num_samples, max_sources) = single(nan);
store.snr_all(num_samples, 1) = single(0);
store.geometry_id_all(num_samples, 1) = single(0);
store.geometry_within_k_all(num_samples, 1) = single(0);
store.sample_within_geometry_all(num_samples, 1) = single(0);
store.error_realization_id_all(num_samples, 1) = single(0);
store.case_repeat_all(num_samples, 1) = single(0);
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

%% Generate one small geometry block at a time
sample_idx = 0;
global_geometry = 0;
error_realization_id = 0;
generation_start = tic;

for K = K_values
    fprintf('Generating K=%d\n', K);
    for geometry = 1:num_geometries_per_k
        global_geometry = global_geometry+1;

        difficulty = sample_discrete_local( ...
            [1, 2, 3], [0.20, 0.40, 0.40]);
        if difficulty == 1
            sep_req = 18; ux_req = 0.10; uz_req = 0.08;
        elseif difficulty == 2
            sep_req = 12; ux_req = 0.055; uz_req = 0.040;
        else
            sep_req = 7; ux_req = 0.025; uz_req = 0.020;
        end
        [az, el] = sample_geometry_local(K, sep_req, ux_req, uz_req);
        if K > 1
            min_sep = minimum_spherical_separation(az, el);
        else
            min_sep = nan;
        end

        % A waveform realization is shared only inside this geometry block.
        [timing_offsets, cfo_values] = sample_sync_parameters( ...
            K, max_timing_offset, cfo_min, cfo_max);
        S = complex(zeros(K, L));
        for source = 1:K
            timing = timing_offsets(source);
            available = L-timing;
            payload = generate_structured_waveform( ...
                available, family_probs, sps_values, ...
                rrc_rolloff_range, rrc_span, cpfsk_h_range, ...
                ofdm_fft_values, ofdm_cp_values, [0, 0], [1, 1]);
            packet = payload;
            packet(1:preamble_len) = shared_preamble;
            n = 0:(available-1);
            phase0 = 2*pi*rand();
            packet = packet.*exp(1j*( ...
                2*pi*cfo_values(source)*n+phase0));
            packet = packet/sqrt(mean(abs(packet).^2)+1e-12);
            S(source, timing+(1:available)) = packet;
        end

        % Exact case-by-SNR balance: for 600 geometries, every case has
        % exactly 100 samples at every configured SNR. Row order is shuffled.
        case_order = case_sequence(randperm(samples_per_geometry));
        snr_sequence = zeros(1, samples_per_geometry);
        for local_sample = 1:samples_per_geometry
            snr_position = mod( ...
                (geometry-1)+case_order(local_sample), ...
                numel(snr_levels))+1;
            snr_sequence(local_sample) = snr_levels(snr_position);
        end
        rows = sample_idx+(1:samples_per_geometry);

        iq_block = zeros( ...
            samples_per_geometry, L, num_ant, 2, 'single');
        source_real_block = zeros( ...
            samples_per_geometry, max_sources, L, 'single');
        source_imag_block = zeros( ...
            samples_per_geometry, max_sources, L, 'single');
        doa_block = nan( ...
            samples_per_geometry, max_sources, 2, 'single');
        timing_block = nan( ...
            samples_per_geometry, max_sources, 'single');
        cfo_block = nan( ...
            samples_per_geometry, max_sources, 'single');
        power_block = nan( ...
            samples_per_geometry, max_sources, 'single');
        snr_block = zeros(samples_per_geometry, 1, 'single');
        geometry_block = single(global_geometry*ones( ...
            samples_per_geometry, 1));
        geometry_within_block = single(geometry*ones( ...
            samples_per_geometry, 1));
        sample_within_block = single((1:samples_per_geometry).');
        realization_block = zeros(samples_per_geometry, 1, 'single');
        repeat_block = zeros(samples_per_geometry, 1, 'single');
        k_block = single(K*ones(samples_per_geometry, 1));
        difficulty_block = single(difficulty*ones( ...
            samples_per_geometry, 1));
        min_sep_block = single(min_sep*ones(samples_per_geometry, 1));
        case_block = zeros(samples_per_geometry, 1, 'single');
        gain_block = zeros(samples_per_geometry, 1, 'single');
        phase_block = zeros(samples_per_geometry, 1, 'single');
        coupling_amp_block = zeros(samples_per_geometry, 1, 'single');
        position_block = zeros(samples_per_geometry, 1, 'single');
        gain_real_block = zeros( ...
            samples_per_geometry, num_ant, 'single');
        gain_imag_block = zeros( ...
            samples_per_geometry, num_ant, 'single');
        coupling_real_block = zeros( ...
            samples_per_geometry, num_ant, num_ant, 'single');
        coupling_imag_block = zeros( ...
            samples_per_geometry, num_ant, num_ant, 'single');
        coords_block = zeros( ...
            samples_per_geometry, 3, num_ant, 'single');

        padded_S = complex(zeros(max_sources, L));
        padded_S(1:K, :) = S;
        source_real_template = single(real(padded_S));
        source_imag_template = single(imag(padded_S));
        case_seen = zeros(1, numel(case_ids));

        for local_sample = 1:samples_per_geometry
            case_id = case_order(local_sample);
            case_position = find(case_ids == case_id, 1, 'first');
            case_seen(case_position) = case_seen(case_position)+1;
            [gain_sigma, phase_sigma, coupling_amp, position_sigma] = ...
                sample_case_parameters_local( ...
                    case_id, case_minimum, case_maximum);

            % New independent calibration realization for every sample.
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

            A = complex(zeros(num_ant, K));
            for source = 1:K
                A(:, source) = steering_local( ...
                    actual_coords, lambda_val, az(source), el(source));
            end
            effective_manifold = coupling*(sensor_gain(:).*A);
            Y_clean = effective_manifold*S;
            signal_power = mean(sum(abs(Y_clean).^2, 1));

            noise = (randn(num_ant, L)+1j*randn(num_ant, L))/sqrt(2);
            noise = noise/sqrt( ...
                mean(sum(abs(noise).^2, 1))+1e-12);
            snr_db = snr_sequence(local_sample);
            noise = noise*sqrt(signal_power/(10^(snr_db/10)));
            Y = Y_clean+noise;

            iq_block(local_sample, :, :, 1) = ...
                reshape(single(real(Y).'), [1, L, num_ant, 1]);
            iq_block(local_sample, :, :, 2) = ...
                reshape(single(imag(Y).'), [1, L, num_ant, 1]);
            source_real_block(local_sample, :, :) = ...
                reshape(source_real_template, [1, max_sources, L]);
            source_imag_block(local_sample, :, :) = ...
                reshape(source_imag_template, [1, max_sources, L]);
            doa_block(local_sample, 1:K, 1) = single(az);
            doa_block(local_sample, 1:K, 2) = single(el);
            timing_block(local_sample, 1:K) = single(timing_offsets);
            cfo_block(local_sample, 1:K) = single(cfo_values);
            power_block(local_sample, 1:K) = 0;
            snr_block(local_sample) = single(snr_db);
            error_realization_id = error_realization_id+1;
            realization_block(local_sample) = ...
                single(error_realization_id);
            repeat_block(local_sample) = ...
                single(case_seen(case_position));
            case_block(local_sample) = single(case_id);
            gain_block(local_sample) = single(gain_sigma);
            phase_block(local_sample) = single(phase_sigma);
            coupling_amp_block(local_sample) = single(coupling_amp);
            position_block(local_sample) = single(position_sigma);
            gain_real_block(local_sample, :) = ...
                single(real(sensor_gain));
            gain_imag_block(local_sample, :) = ...
                single(imag(sensor_gain));
            coupling_real_block(local_sample, :, :) = ...
                reshape(single(real(coupling)), [1, num_ant, num_ant]);
            coupling_imag_block(local_sample, :, :) = ...
                reshape(single(imag(coupling)), [1, num_ant, num_ant]);
            coords_block(local_sample, :, :) = ...
                reshape(single(actual_coords), [1, 3, num_ant]);
        end

        store.IQ_tensors(rows, :, :, :) = iq_block;
        store.source_waveforms_real(rows, :, :) = source_real_block;
        store.source_waveforms_imag(rows, :, :) = source_imag_block;
        store.doas_all(rows, :, :) = doa_block;
        store.timing_offset_all(rows, :) = timing_block;
        store.cfo_all(rows, :) = cfo_block;
        store.source_power_db_all(rows, :) = power_block;
        store.snr_all(rows, :) = snr_block;
        store.geometry_id_all(rows, :) = geometry_block;
        store.geometry_within_k_all(rows, :) = geometry_within_block;
        store.sample_within_geometry_all(rows, :) = sample_within_block;
        store.error_realization_id_all(rows, :) = realization_block;
        store.case_repeat_all(rows, :) = repeat_block;
        store.num_sources_all(rows, :) = k_block;
        store.difficulty_all(rows, :) = difficulty_block;
        store.min_sep_all(rows, :) = min_sep_block;
        store.impairment_case_all(rows, :) = case_block;
        store.gain_sigma_all(rows, :) = gain_block;
        store.phase_sigma_deg_all(rows, :) = phase_block;
        store.coupling_amp_all(rows, :) = coupling_amp_block;
        store.position_sigma_d_all(rows, :) = position_block;
        store.sensor_gain_real(rows, :) = gain_real_block;
        store.sensor_gain_imag(rows, :) = gain_imag_block;
        store.coupling_real(rows, :, :) = coupling_real_block;
        store.coupling_imag(rows, :, :) = coupling_imag_block;
        store.actual_array_coords_all(rows, :, :) = coords_block;
        sample_idx = sample_idx+samples_per_geometry;

        if mod(geometry, 20) == 0 || geometry == num_geometries_per_k
            elapsed = toc(generation_start);
            rate = sample_idx/max(elapsed, 1e-9);
            remaining = (num_samples-sample_idx)/max(rate, 1e-9);
            fprintf(['  K=%d geometry %d/%d | samples %d/%d | ' ...
                'elapsed %.1f min | ETA %.1f min\n'], ...
                K, geometry, num_geometries_per_k, ...
                sample_idx, num_samples, elapsed/60, remaining/60);
        end
    end
end

%% Save scalar configuration and atomically finalize the dataset
store.shared_preamble_real = shared_preamble_real;
store.shared_preamble_imag = shared_preamble_imag;
store.nominal_array_coords = nominal_array_coords;
store.case_ids = case_ids;
store.case_repeats = case_repeats;
store.case_minimum = case_minimum;
store.case_maximum = case_maximum;
store.case_sequence = case_sequence;
store.f_c = f_c;
store.c = c;
store.lambda_val = lambda_val;
store.d = d;
store.L = L;
store.max_sources = max_sources;
store.K_values = K_values;
store.preamble_len = preamble_len;
store.max_timing_offset = max_timing_offset;
store.num_geometries_per_k = num_geometries_per_k;
store.total_geometries = total_geometries;
store.samples_per_geometry = samples_per_geometry;
store.num_samples = num_samples;
store.snr_levels = snr_levels;
store.generator_seed = 16100;
store.dataset_version = single(1610);
store.dataset_role = 'robust_refiner_training';
clear store;

movefile(partial_filename, filename);
fprintf('Done: %s\n', filename);
fprintf('Total generation time: %.1f min\n', toc(generation_start)/60);


function sequence = expand_case_sequence_local(case_ids, repeats)
    sequence = [];
    for position = 1:numel(case_ids)
        sequence = [sequence, repmat( ...
            case_ids(position), 1, repeats(position))]; %#ok<AGROW>
    end
end


function [gain_sigma, phase_sigma, coupling_amp, position_sigma] = ...
        sample_case_parameters_local(case_id, minimums, maximums)
    row = case_id+1;
    value = minimums(row, :)+ ...
        (maximums(row, :)-minimums(row, :)).*rand(1, 4);
    gain_sigma = value(1);
    phase_sigma = value(2);
    coupling_amp = value(3);
    position_sigma = value(4);
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





