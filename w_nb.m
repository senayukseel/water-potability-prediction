data = readtable('water.csv'); %eksik veriler home > clean data kısmından temizlendi

data = sortrows(data,"is_safe","descend"); % 1-0 etiketlerinin sayısını öğrenmek için sıraladık
data(1825:7996, :) = []; %fazla verileri silindi
% 0 etiketi fazla olduğu için toplam veri sayımızı 0 etiketinin sayısını
% göz önünde bulundurarak ayarlandı 

% eşitlenen yeni dataset farklı bir değişkene atanıp kaydedildi.
water='water_equal.csv';
writetable(data, water);
water = readtable('water_equal.csv');

rng('default')% MATLAB'da rastgele sayı üretecinin varsayılan durumuna geri dönmeyi sağlar.  Bu, tekrarlanabilirlik açısından önemlidir.

randomized = water(randperm(size(water, 1)), :);

cv= cvpartition(size(water,1),'Holdout',0.3); 

idxTrain = training(cv);
tblTrain = water(idxTrain,1:end-1);
tblTrainL = water(idxTrain, end);
W = [tblTrain,tblTrainL];

idxTest = test(cv);
tblTest = water(idxTest,1:end-1);
tblTestL= water(idxTest,end);

%train ve test olarak ayırdığım verileri .csv dosyası olarak kaydetme
writetable([tblTrain,tblTrainL], "train.csv") 
writetable([tblTest,tblTestL], "test.csv")


% Naive Bayes sınıflandırıcısı oluşturun
nbMdl = fitcnb(tblTrain(:,1:end-1), tblTrainL.is_safe); % Son sütunun 'is_safe' hedef değişkeni olduğunu varsayalım

% Naive Bayes modelini kullanarak tahmin yapın
pred = predict(nbMdl, tblTest(:,1:end-1)); % Test seti üzerinde tahmin yapma
% Naive Bayes modeli için performans metriklerini hesaplayın


is_safe_double = double(tblTestL.is_safe); %tblTestL veri tipini değiştirilip tahminle geçek sonuçlar kıyaslandı
[pred,is_safe_double] %başta tblTestL table tipinde olduğu için hata verdi.
isCorrect = (pred == is_safe_double);

label = tblTestL.is_safe; % Gerçek sınıf etiketleri
labelPred = pred; % Tahmin edilen sınıf etiketleri


% Karmaşıklık matrisini oluşturun
cm = confusionchart(label, labelPred);

normalized_cm = cm.NormalizedValues; %verilerle yapılan işlemlerin belirli bir ölçekte yapılması için kullanıldı.

% TP, TN, FP, FN değerlerini ekranda gösterme
TP = normalized_cm(1, 1);
FN = normalized_cm(1, 2);
FP = normalized_cm(2, 1);
TN = normalized_cm(2, 2);

po = (TP + TN) / numel(label); % observed proportion
pe = (sum(label == 1) * sum(labelPred == 1) + sum(label == 0) * sum(labelPred == 0)) / numel(label)^2; % expected proportion
po_max = (sum(label == 1) + sum(label == 0)) / numel(label);
kappa = (po - pe) / (po_max - pe);

fprintf('True Positive (TP): %d\n', TP);
fprintf('False Negative (FN): %d\n', FN);
fprintf('False Positive (FP): %d\n', FP);
fprintf('True Negative (TN): %d\n', TN);


% doğruluk (accuracy)
accuracy = (TP + TN) / (TP + TN + FP + FN);

% duyarlılık (sensitivity)
sensitivity = TP / (TP + FN);

% Özgüllük (Specificity)
specificity = TN / (TN + FP);

% Precision ve Recall (Duyarlılık) hesaplama
precision = TP / (TP + FP);
recall = TP / (TP + FN);

% F1 Ölçümü hesaplama
F1_score = 2 * (precision * recall) / (precision + recall);

% % predictedScores adında bir değişken tanımla (örnek olarak rastgele sayılar kullanıldı)
% predictedScores = rand(size(tblTestL.is_safe));


% Gerçek sınıf etiketleri (labels) ve tahmin edilen sınıf skorları (scores) tanımlanır.
labels = tblTestL.is_safe; % Örnek olarak gerçek etiketlerin bulunduğu sütun
%scores = predictedScores; % Örnek olarak tahmin edilen skorlar veya olasılıklar

% ROC eğrisini çizmek için perfcurve kullanılır
[FP,TP,T,AUC] = perfcurve(labels, pred, 1); % '1' sınıfın pozitif etiketi olduğunu belirtir

% ROC eğrisini çizdirme
figure;
plot(FP, TP)
hold on 
plot([0,1],[0,1],'k--') % diagonal çizgi)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve')

% AUC hesaplama
AUC = trapz(FP, TP);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Eğrisi Naive Bayes');
disp("AUC:"+ AUC);

fprintf("Accuracy: %4f\n", accuracy);
fprintf('Kappa Değeri: %.4f\n', kappa);
fprintf('Sensitivity: %.4f\n', sensitivity);
fprintf('Specificity: %.4f\n', specificity);
fprintf('Precision: %.4f\n' , precision);
fprintf('Recall: %4f\n', recall);
fprintf('F-Ölçümü: %4f\n', F1_score);

