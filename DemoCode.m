%% Demo

% Adaptive Filtering

% May 23, 2019

%--------------------------------------------------------------------------
% Loading and listening to original noisy signal
%--------------------------------------------------------------------------
[n, ~]=audioread('talkingNoise.mp3'); % Interfering Signal (ref)
[x, fs]=audioread('bassLineTalkingNoise.mp3'); % Recorded Signal (Static recording)

x = x(:, 1);
n = n(:, 1);

sound(x, fs)
%%
%--------------------------------------------------------------------------
% LMS Filter
%--------------------------------------------------------------------------
clc;
clear;

[x, ~]=audioread('talkingNoise.wav'); % Interfering Signal (ref)
[y, fs]=audioread('bassLineTalkingNoise.wav'); % Recorded Signal (Static recording)
[yclean, ~] = audioread('Clean bass.wav'); % Recorded Signal (Static recording)

x = x(:,1);
y = y(:,1);
yclean = yclean(:,1);
 
%We check there is no offset between the noise and the corrupted signal
c = abs(xcorr(x,y));
L = max(length(y), length(x));
figure;
plot(-L+1:L-1, c);
title('Autocorrelation between noise and noisy signal');
xlabel('samples');

%See if there is offset with the clean signal
c = abs(xcorr(y,yclean));
L = max(length(y), length(yclean));
figure;
plot(-L+1:L-1, c);
title('Autocorrelation between noisy and clean signal');
xlabel('samples');
ix = find(max(c)==c) - L;
yclean_corrected = zeros(length(yclean), 1); 
yclean_corrected(ix:end) = yclean(1: end - ix+1); 
y = y(1:length(yclean_corrected));
x = x(1:length(yclean_corrected));
yclean = yclean_corrected;

%yp will be the signal after we have removed the noise through adaptive
%filtering
yp=zeros(length(y),1);
mu=0.003;
Nf = 800;

fad=zeros(Nf,1);

Nit=20;
tic;
%Adaptive filtering
for i=Nf+1:length(y)
    X=x(i-Nf+1:i);
    for j=1:Nit
        fad=fad+mu*X*(y(i)-X'*fad); %LMS
    end
    yp(i)=y(i)-fad'*X;
    
    % Plot the adaptive filter at some instants
    if((mod(i-2*fs,5*fs)==0) && (i>10*fs))
        figure;
        plot((1:Nf)/fs,fad(end:-1:1));
        xlabel('Time [ms]');
        ylabel('Amplitude');
        L=max(abs(fad));
        ylim([-1.5*L 1.5*L]);
        title(sprintf('Estimation Room Impluse Response at t=%d s',i/fs));
    end
end

toc;
t = toc;

disp('Execution time for LMS: ');
disp(t);

%Let's plot some results
figure; 
plot((0:length(yp)-1)/fs,(y-yp).^2)
title('(y-yp).^2');
ylabel('Error power');
xlabel('time (s)');
fad=fad(end:-1:1);

figure;
plot(y,'b');hold on; plot(yp,'r');
xlabel('Samples');
ylabel('Amplitude');
title('Sound Wave A');
legend('Original','Filtered');

filename = 'lms.wav';
audiowrite(filename,yp,fs);

%Let's listen to the cleaned signal
sound(yp, fs);

%%
%--------------------------------------------------------------------------
% NLMS Filter
%--------------------------------------------------------------------------
yp=zeros(length(y),1);
munlsm = 0.003*Nf*mean(x.^2);% to have on average similar learning rate compared to LMS

Nf = 800;

fad=zeros(Nf,1);

%for numerical stability purposes
epsilon = 1e-5;

tic; 
% Adaptive filtering
for i=Nf:length(y)
    X=x(i-Nf+1:i);
    fad=fad+munlsm*X*(y(i)-X'*fad)/(sum(X.^2)+ epsilon); %NLMS
    yp(i)=y(i)-fad'*X;
    
    % Plot the adaptive filter at some instants
    if((mod(i-2*fs,5*fs)==0) && (i>10*fs))
        figure;
        plot((1:Nf)/fs,fad(end:-1:1));
        xlabel('Time [ms]');
        ylabel('Amplitude');
        L=max(abs(fad));
        ylim([-1.5*L 1.5*L]);
        title(sprintf('Estimation Room Impluse Response at t=%d s',i/fs));
    end
end
toc; 
t = toc; 
disp('Execution time for NLMS: ');
disp(t);
fad=fad(end:-1:1);
figure;
plot(y,'b');hold on; plot(yp,'r');
xlabel('Time [ms]');
ylabel('Amplitude');
title('Sound Wave A');
legend('Original','Filtered');

filename = 'nlms.wav';
audiowrite(filename,yp,fs);

figure;
plot((0:length(yp)-1)/fs,(y-yp).^2)
title('(y-yp).^2');
ylabel('Error power');
xlabel('time (s)');
ylim([0,6]);

sound(yp, fs)

%%
%--------------------------------------------------------------------------
% RLS Filter
%--------------------------------------------------------------------------
% Filter Parameters
p       = 800;                % filter order
lambda  = 0.985;              % forgetting factor
laminv  = 1/lambda;
delta   = 1.0;              % initialization parameter
% Filter Initialization
w       = zeros(p,1);       % filter coefficients
P       = delta*eye(p);     % inverse correlation matrix
e       = x*0;              % error signal
for m = p:length(x)
    % Acquire chunk of data
    y = n(m:-1:m-p+1);
    % Error signal equation
    e(m) = x(m)-w'*y;
    % Parameters for efficiency
    Pi = P*y;
    % Filter gain vector update
    k = (Pi)/(lambda+y'*Pi);
    % Inverse correlation matrix update
    P = (P - k*y'*P)*laminv;
    % Filter coefficients adaption
    w = w + k*e(m);
    % Counter to show filter is working

end
% Plot RLS filter results
t = linspace(0,length(x)/fs,length(x));
figure;
plot(t,x,t,e);
% Plot comparison of results to original signal (only for generated data)

figure;
plot(t,s,t,e);
title('Result of RLS Filter')
xlabel('Time (s)');
legend('Signal', 'Filtered', 'Location', 'NorthEast');
title('Comparison of Filtered Signal to Original Signal');

% Calculate SNR improvement
SNRi = 10*log10(var(x)/var(e));
disp([num2str(SNRi) 'dB SNR Improvement'])

sound(e, fs)

%%
%--------------------------------------------------------------------------
% Offline LMS
%--------------------------------------------------------------------------
clc;
clear all;
close all

[x, ~]=audioread('talkingNoise.wav'); % Interfering Signal (ref)
[y, fs]=audioread('bassLineTalkingNoise.wav'); % Recorded Signal (Static recording)
x = x(:,1);
y = y(:,1);

yp=zeros(length(y),1);
mu=0.0003;
Npast = 100;
Nfuture = 100;
fad=zeros(Npast+Nfuture,1);
Nit=20;

% Adaptive filtering
for i=Npast:(length(y)-Nfuture)
    X=x(i-Npast+1:(i+Nfuture));
    for j=1:Nit
        fad=fad+mu*X*(y(i)-X'*fad); %LMS
    end
    yp(i)=y(i)-fad'*X;
    
    % Plot the adaptive filter at some instants
    if((mod(i-2*fs,5*fs)==0) && (i>10*fs))
        figure;
        plot((1:(Npast+Nfuture))/fs,fad(end:-1:1));
        xlabel('Time [ms]');
        ylabel('Amplitude');
        L=max(abs(fad));
        ylim([-1.5*L 1.5*L]);
        title(sprintf('Estimation Room Impluse Response at t=%d s',i/fs));
    end
end

figure;
plot((0:length(yp)-1)/fs,(y-yp).^2)
title('(y-yp).^2');
ylabel('Error power');
xlabel('time (s)');

figure;
plot(y,'b');hold on; plot(yp,'r');
xlabel('Time [ms]');
ylabel('Amplitude');
title('Sound Wave A');
legend('Original','Filtered');

filename = 'lms-offline.wav';
audiowrite(filename,yp,fs);

figure
plot((y-yp).^2)
ylim([0 1]);
ylabel('Error power');
xlabel('time (s)');

