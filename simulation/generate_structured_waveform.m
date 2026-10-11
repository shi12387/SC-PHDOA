function s = generate_structured_waveform(L, family_probs, sps_values, ...
    rolloff_range, rrc_span, h_range, fft_values, cp_values, cfo_range, duty_range)
% Toolbox-free structured complex-baseband waveform generator.
family = sample_local(1:3, family_probs);
switch family
    case 1
        sps = sps_values(randi(numel(sps_values)));
        beta = rolloff_range(1) + diff(rolloff_range) * rand();
        order_set = [4, 8, 16];
        M = order_set(randi(numel(order_set)));
        nsym = ceil((L + 2 * rrc_span * sps) / sps) + 4;
        idx = randi([0, M - 1], 1, nsym);
        if M == 16
            ii = 2 * mod(idx, 4) - 3;
            qq = 2 * floor(idx / 4) - 3;
            symbols = (ii + 1j * qq) / sqrt(10);
        else
            symbols = exp(1j * 2 * pi * idx / M);
        end
        up = zeros(1, nsym * sps);
        up(1:sps:end) = symbols;
        pulse = local_rrc(beta, rrc_span, sps);
        tmp = conv(up, pulse, 'same');
    case 2
        sps = sps_values(randi(numel(sps_values)));
        h = h_range(1) + diff(h_range) * rand();
        nsym = ceil(L / sps) + 4;
        bits = 2 * randi([0, 1], 1, nsym) - 1;
        freq = repelem(bits, sps);
        smooth_len = max(3, 2 * sps + 1);
        g = exp(-0.5 * ((-(smooth_len-1)/2:(smooth_len-1)/2) / (0.45*sps)).^2);
        g = g / sum(g);
        freq = conv(freq, g, 'same');
        tmp = exp(1j * pi * h / sps * cumsum(freq));
    otherwise
        nfft = fft_values(randi(numel(fft_values)));
        cp_frac = cp_values(randi(numel(cp_values)));
        ncp = max(1, round(nfft * cp_frac));
        nblocks = ceil(L / (nfft + ncp)) + 1;
        tmp = complex([]);
        active = [3:(nfft/2-1), (nfft/2+2):(nfft-2)];
        for b = 1:nblocks
            bins = complex(zeros(1, nfft));
            q = randi([0, 3], 1, numel(active));
            bins(active) = exp(1j * (pi/4 + pi/2*q));
            symbol = ifft(bins) * sqrt(nfft);
            tmp = [tmp, symbol(end-ncp+1:end), symbol]; %#ok<AGROW>
        end
end

if numel(tmp) < L
    tmp = repmat(tmp, 1, ceil(L / numel(tmp)));
end
start_max = max(1, numel(tmp) - L + 1);
start = randi(start_max);
s = tmp(start:start+L-1);

% Random burst envelope, timing-independent of all labels.
duty = duty_range(1) + diff(duty_range) * rand();
active_len = max(64, min(L, round(duty * L)));
offset = randi([0, L - active_len]);
envelope = zeros(1, L);
edge = min(16, floor(active_len / 4));
envelope(offset + (1:active_len)) = 1;
if edge > 1
    ramp = sin(linspace(0, pi/2, edge)).^2;
    envelope(offset + (1:edge)) = ramp;
    envelope(offset + active_len - edge + (1:edge)) = fliplr(ramp);
end
s = s .* envelope;

cfo = cfo_range(1) + diff(cfo_range) * rand();
phase0 = 2 * pi * rand();
n = 0:(L-1);
s = s .* exp(1j * (2*pi*cfo*n + phase0));
s = s / sqrt(mean(abs(s).^2) + 1e-12);
end

function h = local_rrc(beta, span, sps)
t = (-span*sps:span*sps) / sps;
h = zeros(size(t));
for i = 1:numel(t)
    x = t(i);
    if abs(x) < 1e-10
        h(i) = 1 + beta * (4/pi - 1);
    elseif beta > 0 && abs(abs(4*beta*x) - 1) < 1e-8
        h(i) = beta/sqrt(2) * ((1+2/pi)*sin(pi/(4*beta)) + ...
            (1-2/pi)*cos(pi/(4*beta)));
    else
        h(i) = (sin(pi*x*(1-beta)) + 4*beta*x*cos(pi*x*(1+beta))) / ...
            (pi*x*(1-(4*beta*x)^2));
    end
end
h = h / sqrt(sum(h.^2) + 1e-12);
end

function value = sample_local(values, probs)
cdf = cumsum(probs(:) / sum(probs));
value = values(find(rand() <= cdf, 1, 'first'));
end

