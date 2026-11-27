function generate_round3_cdl(som, output, count)
% Draw 3GPP CDL-C phases in QuaDRiGa and sample both speeds on one track.
addpath(fullfile(som, 'tools', 'QuaDRiGa', 'quadriga_src'));
if ~isfolder(output), mkdir(output); end
rng(729041,'twister');
fc = 3.5e9;
light_speed = 299792458;
snapshot_s = 0.5e-3;
spacing_hz = 180e3;
peak_doppler_hz = [40, 140];
rms_delay_ns = [50, 200];
velocity = peak_doppler_hz * light_speed / fc;
% A shared dense track yields identical scaling, antenna patterns and phases.
low_steps = 2 * (0:15);
high_steps = 7 * (0:15);
positions = (0:high_steps(end)) * velocity(1) * snapshot_s / 2;
transmitter = qd_arrayant('omni');
transmitter.copy_element(1, 2:32);
[z, y] = ndgrid(0:3, 0:7);
transmitter.element_position = light_speed / (2 * fc) * [zeros(1,32); y(:)'; z(:)'];
H = complex(zeros(count, 4, 16, 32, 32, 'single'));
profile_rms_ns = zeros(count,2);
for draw = 1:count
    builder = qd_builder.gen_cdl_model('NR-CDL-C', fc, 0, 0, 50);
    builder.tx_array = transmitter;
    track = builder.rx_track;
    track.positions = [positions; zeros(2, numel(positions))];
    track.movement_profile = [0, 15 * snapshot_s; 0, positions(end)];
    response = builder.get_channels;
    assert(size(response.coeff,2)==32 && size(response.coeff,4)==numel(positions));
    coeff = response.coeff;
    keep = squeeze(any(any(abs(coeff) > 0, 4), 2));
    coeff = coeff(:,:,keep,:);
    taps = response.delay(keep,1);
    powers = builder.pow(builder.pow > 0);
    assert(numel(taps) == numel(powers));
    powers = powers(:) / sum(powers);
    taps = taps - min(taps);
    taps = taps / sqrt(sum(powers .* (taps - sum(taps .* powers)).^2));
    % The same scale is used in all cells. The unmodified phases remain paired.
    coeff = coeff / sqrt(mean(sum(abs(coeff).^2, 3), 'all'));
    for delay = 1:2
        tau = taps * rms_delay_ns(delay) * 1e-9;
        profile_rms_ns(draw,delay) = 1e9 * sqrt(sum(powers .* (tau-sum(tau.*powers)).^2));
        phase = exp(-2i*pi*tau*((0:31)*spacing_hz));
        for speed = 1:2
            steps = low_steps;
            if speed==2, steps = high_steps; end
            coefficients = permute(coeff(:,:,:,steps+1), [4,2,3,1]);
            transfer = reshape(reshape(coefficients,[],numel(taps)) * phase, [16,32,32]);
            H(draw, (delay-1)*2+speed, :, :, :) = single(transfer);
        end
    end
end
assert(max(abs(H(:,1,1,:,:)-H(:,2,1,:,:)), [], 'all') < 1e-6);
assert(max(abs(H(:,1,:,:,1)-H(:,3,:,:,1)), [], 'all') < 1e-6);
assert(max(abs(profile_rms_ns-rms_delay_ns), [], 'all') < 1e-8);
matlab_version=version;
quadriga_version=qd_simulation_parameters.version;
save(fullfile(output,'cdl_c_raw.mat'), 'H', 'peak_doppler_hz', 'rms_delay_ns', 'profile_rms_ns', ...
     'fc', 'velocity', 'spacing_hz', 'snapshot_s', 'matlab_version', 'quadriga_version', '-v7');
fprintf('QuaDRiGa CDL-C: %d draws, %s\n', count, mat2str(size(H)));
end
