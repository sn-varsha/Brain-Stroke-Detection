% Adım 0: Orijinal görüntüyü oku
img = imread('16681.png'); % kendi dosyan

% Gri tonlama (zaten gri olabilir ama garanti olsun)
if size(img, 3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end

% Adım 1: 245 ve üstü piksel değerlerini sıfır yap (kemik ve çeper temizliği)
threshold_value = 245;
gray(gray >= threshold_value) = 0;

% Adım 2: Median filtre uygula (pepper noise temizliği)
gray_filtered = medfilt2(gray, [5 5]);

% Adım 3: Küçük beyaz lekeleri silmek için erosion uygula
se_erode = strel('disk', 4); % 4 yarıçaplı disk ile erosion
eroded = imerode(gray_filtered, se_erode);

% Adım 4: Histogram eşitleme (kontrast artırma)
gray_eq = adapthisteq(eroded);

% Adım 5: Thresholding ile Mask oluştur
mask_threshold = 191; % Eşik değeri
mask = gray_eq > mask_threshold; % 191'den büyük parlaklıklar 1 olacak

% Adım 6: Closing uygula (beyaz boşlukları fiziksel kapatmak için)
se_close = strel('disk', 8); % 5 yarıçaplı bir disk ile closing
mask_closed = imclose(mask, se_close);

% Sonuçları göster
figure;
subplot(2,3,1); imshow(gray, []); title('Kemik Temizlenmiş');
subplot(2,3,2); imshow(gray_filtered, []); title('Median Filtreli');
subplot(2,3,3); imshow(eroded, []); title('Erosion Uygulanmış');
subplot(2,3,4); imshow(gray_eq, []); title('Histogram Eşitlemesi Yapılmış');
subplot(2,3,5); imshow(mask); title('Ham MASK (Threshold sonrası)');
subplot(2,3,6); imshow(mask_closed); title('Closing Yapılmış MASK (Boşluklar Kapatıldı)');
