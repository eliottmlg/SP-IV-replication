clear
load('C:\Users\eliot\Downloads\118082-V1\online_appendix\results_var/MBC.mat')

aaaIRF = mirf(2,:,:)
aaaIRF = reshape(aaaIRF,120,6)
save("C:\Users\eliot\Documents\REPOSITORIES\SP-IV-replication\data/aaaIRF.mat","aaaIRF")
writematrix(aaaIRF,'C:\Users\eliot\Documents\REPOSITORIES\SP-IV-replication\data/aaaIRF.csv')

aaaFEV = mfev(2,:,:)
aaaFEV = reshape(aaaFEV,200,6)
save("C:\Users\eliot\Documents\REPOSITORIES\SP-IV-replication\data/aaaFEV.mat","aaaFEV")
writematrix(aaaIRF,'C:\Users\eliot\Documents\REPOSITORIES\SP-IV-replication\data/aaaFEV.csv')

MBC_shock = Qmat(2,:,:)
MBC_shock = median(MBC_shock,2)
save("C:\Users\eliot\Documents\REPOSITORIES\SP-IV-replication\data/MBC_shock.mat","MBC_shock")
writematrix(MBC_shock,'C:\Users\eliot\Documents\REPOSITORIES\SP-IV-replication\data/MBC_shock.csv')
