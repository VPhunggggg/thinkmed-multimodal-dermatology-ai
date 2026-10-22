from classifier import SkinClassifier
clf = SkinClassifier()

# Ảnh da lành tải từ mạng, hoặc ảnh chụp bất kỳ
res = clf.detect("test_image/psoriasis_5.jpg")
print(res.reason, res.probs)
