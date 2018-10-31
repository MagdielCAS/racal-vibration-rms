close all;
clear
clc

%% Constants
g = 9.807;
dataNum = 85;
files = 512; %510, the dir command consider '.' and '..' as files
startHour = 2795;

%% Variables
data = zeros(dataNum,1);
t = zeros(dataNum,1);

%% Load data
list = dir('/home/magdiel/Documentos/TCC/data/vib_25h_RACAL/'); 
cont = 1;
for i = 3:files %starts at 3 to disconsider '.' and '..' 
   name = list(i).name;
   split = strsplit(name, 'h');
   if strcmp(split(2),'_Track1.mat')
        temp = load(strcat('/home/magdiel/Documentos/TCC/data/vib_25h_RACAL/',name));
        data(cont,1) = rms(temp.X);
        t(cont,1) = str2double(split(1)) - startHour;
        cont = cont + 1;
   end
end

%% Plot data

figure;
plot(t,data,'*')
