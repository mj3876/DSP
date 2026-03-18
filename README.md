초기 성능
<img width="982" height="849" alt="스크린샷 2026-03-16 182913" src="https://github.com/user-attachments/assets/d2da028a-57ff-4ac6-b326-ae5ace3e6b3b" />

---

# 🎙️ 화자 분류(Speaker Classification) 모델 실험 일지_최종

이 문서는 40명의 화자를 분류하는 모델의 성능 최적화를 위해 시도한 다양한 학습 기법 중, **최종적으로 채택된 기법**과 실험 후 **기각된 기법**들의 내역과 그 이유를 정리한 것입니다.

---

## 채택한 기법 (Adopted Methods)

### 1. Label Smoothing (`smoothing=0.1`)
* **설명:** 일반적인 Cross Entropy Loss는 정답 클래스에 100% 확률을 부여하지만, Label Smoothing은 정답 클래스에 90%, 나머지 39개 클래스에 10%의 확률을 균등하게 분배합니다.
* **채택 이유:**
    * 모델이 자신의 예측에 과도하게 확신(Overconfidence)하는 것을 방지합니다.
    * 비슷한 음성 특성을 가진 화자들 사이의 혼동을 부드럽게 처리하여 학습을 안정화합니다.
    * 과적합(Overfitting)을 방지하고 모델의 일반화 성능을 향상시킵니다.

### 2. Time Masking (`max=20`, 1회)
* **설명:** SpecAugment 기법의 일환으로, Mel-spectrogram의 시간 축에서 랜덤한 구간(최대 20 프레임)을 0으로 마스킹합니다.
* **채택 이유:**
    * 모델이 특정 시간 구간의 특징에만 의존하지 않고, 전체적인 음성 패턴을 학습하도록 유도합니다.
    * 실제 환경에서 발생할 수 있는 순간적인 잡음이나 오디오 끊김 현상에 대한 강건성(Robustness)을 높입니다.
    * Frequency Masking과 달리, 화자 고유의 핵심 주파수 특성(포먼트, 피치 등)을 온전히 보존할 수 있습니다.

### 3. 3-Model Ensemble (`seed=42, 123, 777`)
* **설명:** 동일한 모델 구조를 서로 다른 랜덤 시드(Random Seed)로 3번 학습시킨 후, 추론(Inference) 단계에서 3개 모델의 출력값(Logits)을 평균 내어 최종 예측을 수행합니다.
* **채택 이유:**
    * 각 시드마다 모델이 서로 다른 Local Optimum에 수렴하므로, 다각적인 관점의 예측 결과를 얻을 수 있습니다.
    * 개별 모델이 범할 수 있는 우연한 실수를 다른 모델들이 상호 보완해 줍니다.
    * 예측 결과의 분산을 줄여주어 최종 성능을 안정적으로 확보할 수 있습니다.

### 4. Test-Time Augmentation (TTA, `10 crops`)
* **설명:** 테스트(추론) 시 단일 샘플에서 랜덤하게 10개의 서로 다른 구간을 잘라내어(Crop) 각각 예측을 수행한 뒤, 그 결과를 평균 냅니다.
* **채택 이유:**
    * 학습 단계와 동일하게 테스트 단계에서도 다양한 시간 구간의 정보를 활용할 수 있습니다.
    * 특정 단일 Crop에서 발생할 수 있는 우연한 예측 오류를 평균을 통해 상쇄합니다.
    * 별도의 추가 학습 없이 추론 단계의 기법 변경만으로 성능 향상을 이끌어냅니다.

### 5. CosineAnnealingLR (`T_max=30, eta_min=1e-6`)
* **설명:** 학습률(Learning Rate)을 코사인 곡선 형태로 점진적이고 부드럽게 감소시킵니다. (기존: 4 Epoch마다 절반으로 감소하는 StepLR 사용)
* **채택 이유:**
    * 학습 후반부로 갈수록 더 세밀한 가중치 업데이트가 가능하여 최적점에 안정적으로 수렴합니다.
    * 급격한 학습률 변화로 인해 발생할 수 있는 학습 불안정 현상을 방지합니다.
    * 데이터나 Augmentation 파이프라인을 수정하지 않는 안전한 변경 사항입니다.
    * 이미 99%+ 이상의 높은 성능에 도달한 모델을 미세 조정(Fine-tuning)하는 데 매우 유리합니다.

---

## 기각된 기법 (Discarded Methods)

### 1. SpecAugment 강화 (Frequency Masking 추가)
* **설명:** 기존의 Time Masking에 더해 주파수 축을 가리는 Frequency Masking을 추가하고, 각각 2회씩 적용하여 Augmentation의 강도를 높였습니다.
* **기각 이유:**
    * **정체성 훼손:** 화자 식별 문제에서 주파수 정보(포먼트, 피치 등)는 "누구의 목소리인가"를 결정짓는 핵심 특징입니다. 이를 마스킹하는 것은 데이터의 본질적 특성을 손상시킵니다.
    * **성능 하락:** 실험 결과, 정확도가 99.35%에서 98% 초반대로 오히려 하락하여 기각했습니다.

### 2. Mixup (`alpha=0.4, prob=0.5`)
* **설명:** 두 개의 학습 음성 데이터(예: 화자 A 60% + 화자 B 40%)를 일정 비율로 혼합하여 새로운 가상의 데이터를 생성하는 기법입니다.
* **기각 이유:**
    * **정답의 모호성:** 화자 식별 태스크의 특성상 음성이 섞이면 정답 레이블 자체가 불명확해져 모델의 혼란을 가중시킵니다.
    * **데이터셋 규모:** 확보된 데이터가 3,718개로 비교적 적은 상황에서 너무 강한 Augmentation은 오히려 학습을 방해했습니다.
    * **성능 하락:** 실험 결과, 정확도가 98% 후반대로 하락하여 최종 파이프라인에서 제외했습니다.

---[test_predictions (1).csv](https://github.com/user-attachments/files/26076226/test_predictions.1.csv)
path,speaker_id
27340950127c453b.wav,speaker_40
1dc3363622ea457a.wav,speaker_40
859de29c4904452e.wav,speaker_20
6bde61d99b0a4134.wav,speaker_20
a9ce5effda6b46a0.wav,speaker_01
d354d3a790af43c7.wav,speaker_19
5c8b751a1aa5468c.wav,speaker_32
6e019904b7f043e4.wav,speaker_09
e56f645a06444ee6.wav,speaker_08
a4264ac117cd4610.wav,speaker_28
a57c93563ad94ce8.wav,speaker_36
5ab20b403cf24e74.wav,speaker_29
b5de02bd22034146.wav,speaker_05
d64a3f2af4ab476b.wav,speaker_30
095148f4a04146aa.wav,speaker_07
02db0f187e624cf9.wav,speaker_33
1eafaa5868504169.wav,speaker_35
13d26e19c98749d3.wav,speaker_04
62313c0a28a047c9.wav,speaker_34
978411c5aa82401f.wav,speaker_24
48e1322d6ca540ec.wav,speaker_39
7afa5a1b3fe14c90.wav,speaker_09
14c2c12e68974618.wav,speaker_11
8e4dc824920b48bb.wav,speaker_17
62a060dbc3504b7b.wav,speaker_31
1551e0402fa941e9.wav,speaker_34
062591fadf194d45.wav,speaker_15
0dca6459330847c6.wav,speaker_30
6ec62a5295784dbb.wav,speaker_25
3e18c14efa5645c8.wav,speaker_19
ea34ac9e789344ab.wav,speaker_40
7c03954cddbf45ce.wav,speaker_04
3bdfa8d3442e4d6e.wav,speaker_10
770069388db14c09.wav,speaker_11
487162a424e047ae.wav,speaker_18
60712fd517ee4fbb.wav,speaker_11
758d0f2ba2314083.wav,speaker_33
6acaf5daebea4198.wav,speaker_18
37d3a06f06f64319.wav,speaker_20
459dd1ac9ad0423e.wav,speaker_21
4d2f3eb1fdfd4107.wav,speaker_37
862953ae7da0488b.wav,speaker_16
74334f8f90a34598.wav,speaker_27
bcc6b415668e4b47.wav,speaker_19
e8f0f917184b4e6b.wav,speaker_15
a6ef07a668194f87.wav,speaker_03
ad3b7a101428420f.wav,speaker_22
7d7b1eeacf59413f.wav,speaker_21
7ac438fafb804f11.wav,speaker_13
ab2d0c9a08ab418c.wav,speaker_39
91483b51cc6941d6.wav,speaker_38
3313910e42a941a4.wav,speaker_30
551507f807d14334.wav,speaker_40
3a1f5598aa3440be.wav,speaker_17
cfabd7965ba14630.wav,speaker_14
fd5cd23dda344714.wav,speaker_40
6441bae8c188445f.wav,speaker_31
3a95c4b0185f4c70.wav,speaker_12
27bd886978c74603.wav,speaker_06
52621ccffa014e06.wav,speaker_26
931d946cb3cb4eb3.wav,speaker_08
7d75bac7e7e143d6.wav,speaker_01
3c13448d82dc4e73.wav,speaker_31
1d5e261ab67946c2.wav,speaker_32
587137084aa54797.wav,speaker_20
b534312144504cc3.wav,speaker_10
69d3ab31c4cf4524.wav,speaker_24
7ac8a9fafc1149d6.wav,speaker_28
57ee52d424dc4221.wav,speaker_14
c6263a8aee4a438c.wav,speaker_02
a9bd98d5f0334b6d.wav,speaker_18
c8669fb88da34778.wav,speaker_02
b5132cbc278e4dfa.wav,speaker_38
6f07571b771c4c17.wav,speaker_40
223888963bb046a5.wav,speaker_03
a5c99f20404244d8.wav,speaker_23
6afa3b3cf51a480e.wav,speaker_28
f924e874aea6498b.wav,speaker_15
4d5f373a764b4182.wav,speaker_20
74d32744e9fb4ecc.wav,speaker_12
95919cb2387f4a8d.wav,speaker_01
6696103309584f99.wav,speaker_25
6ecd566365b04077.wav,speaker_38
5730bad47bb34e42.wav,speaker_33
9137737a9ba34663.wav,speaker_02
0e6a4add708c43d7.wav,speaker_11
690fe21dae604e8a.wav,speaker_29
68a6574e32874d84.wav,speaker_29
2c9fdf847153406c.wav,speaker_26
9afe4bfc7b434d05.wav,speaker_31
6ae4f41c78664e31.wav,speaker_02
b9ffae480db14cb6.wav,speaker_17
f1c5d26b1b4f4fdc.wav,speaker_36
a564f39ccbdf4b8e.wav,speaker_10
1656754c4447496b.wav,speaker_18
d539e5a8619d4eb2.wav,speaker_03
d74591b4aa1d40c6.wav,speaker_30
7df95b271a9e4a6d.wav,speaker_12
be8baa3d6c8d4cf6.wav,speaker_08
d95a7ed3a3a74640.wav,speaker_07
8da58aafba3341e4.wav,speaker_37
1203f0c9c2384501.wav,speaker_28
d8d8c2aa07ff4e1b.wav,speaker_37
d388b95c1ece4448.wav,speaker_11
3275d91611b14d16.wav,speaker_10
7b9ca43a2eee499d.wav,speaker_19
7cadf985c8134ec3.wav,speaker_31
9b5f4ccd6e604ae9.wav,speaker_31
9878415204334abe.wav,speaker_28
49a17b0422634061.wav,speaker_14
777020d3969b495c.wav,speaker_32
290b79eccc5c4788.wav,speaker_29
a43af3febf3b4495.wav,speaker_29
7608add2f4564ac5.wav,speaker_35
682ffec1251f4446.wav,speaker_31
23d156e6513942c3.wav,speaker_14
5525b491a38243e5.wav,speaker_13
f5b0dc4411bf4639.wav,speaker_32
b719a26c8c754247.wav,speaker_08
ebb44de9a2bc4cf5.wav,speaker_40
277b0271b38d4732.wav,speaker_07
0c1be5dc6cc24623.wav,speaker_05
27d9bdfc27d24ce5.wav,speaker_03
2062fa5961a844b4.wav,speaker_13
172920edce3b4ea1.wav,speaker_38
335fbf6d53ec40d6.wav,speaker_21
45e4c01b2b40445a.wav,speaker_18
be234ddb97974fa1.wav,speaker_22
3557d6b279454959.wav,speaker_12
d301205f85b24ef9.wav,speaker_24
d49c35b1879f475f.wav,speaker_23
cd5303945f084cbd.wav,speaker_21
7aa29cd043574653.wav,speaker_30
e676ba6245de4ed5.wav,speaker_06
5cf9603704a5479f.wav,speaker_23
23cfdc44aedb4e9b.wav,speaker_28
c432c810a3fe4f39.wav,speaker_10
3b783ecdbe3e4403.wav,speaker_38
1781a7379a304aaf.wav,speaker_23
b83acdaf964c496c.wav,speaker_22
65786056e9194e90.wav,speaker_28
7fa32b38e18e4aba.wav,speaker_02
5c5d490b4e904d50.wav,speaker_40
eb370d1bbd8642ea.wav,speaker_13
e48c5c019a7e42d0.wav,speaker_23
8c9955d2ad9d44ff.wav,speaker_25
f2ebc40fcc1245bc.wav,speaker_07
c5d9a78d24b54687.wav,speaker_06
be06f91b85a0404a.wav,speaker_10
accfb92ddc9243be.wav,speaker_08
5711642859604c33.wav,speaker_37
033bdaea087a4f76.wav,speaker_35
b1bc608f81e3426a.wav,speaker_33
724ccefe2a5445c8.wav,speaker_38
e0fa69e88fb94174.wav,speaker_32
528ad4837a0240a7.wav,speaker_40
13b6d7d8037e404c.wav,speaker_27
20597fe01ea24f75.wav,speaker_15
db8585b07c2f4dcc.wav,speaker_33
a43d1b65c8414ab6.wav,speaker_07
92e053554c074251.wav,speaker_01
34d9f5d270f140c2.wav,speaker_16
0c46c9023b894ae5.wav,speaker_30
2d64ae6deeab4558.wav,speaker_30
73531128d3864358.wav,speaker_32
7a822386f8294982.wav,speaker_26
5111fe6f65ad4fce.wav,speaker_38
dca20c1e40a7495f.wav,speaker_24
4cf2eb7f0cf24944.wav,speaker_12
203e8b4f9fe34245.wav,speaker_16
8102f98f9f6346e2.wav,speaker_02
3a2507693d7a4cab.wav,speaker_03
7b07bdac334b42d6.wav,speaker_06
83d20c3f6f164e72.wav,speaker_37
8d8598f7500f4d40.wav,speaker_20
bc1ad38d1ad34d15.wav,speaker_31
615b493cde9f4271.wav,speaker_15
57d9894053574e2f.wav,speaker_14
1978f26336fb42c2.wav,speaker_09
1ea94dca78694128.wav,speaker_28
062b8b744f724229.wav,speaker_39
e1e86fe4179b49b5.wav,speaker_22
220c6d18a1ff4db6.wav,speaker_20
6e2aa22e84a94a04.wav,speaker_24
c5b3eb317896412d.wav,speaker_02
4c3c2a8c3d444fd7.wav,speaker_15
cac6dcba75ae43cf.wav,speaker_05
f6d1a12c60b549f1.wav,speaker_17
b0f3e17987d14377.wav,speaker_25
7098da0225644333.wav,speaker_09
0b8edc8f359a4d97.wav,speaker_08
8bd6336090b541eb.wav,speaker_15
c0f76512c7c64e4a.wav,speaker_32
dd7133bbaa8f47cd.wav,speaker_13
6955be9687b24317.wav,speaker_01
891282630f13442d.wav,speaker_03
0e48fc64cbfa46f9.wav,speaker_08
2721607891314d39.wav,speaker_32
b9eb867720d84e09.wav,speaker_22
d5480c2bffc2470f.wav,speaker_30
d425174f5f2e4616.wav,speaker_05
b9e7480cf6aa45aa.wav,speaker_30
843cb5eadf074b24.wav,speaker_17
0171ab4aa1594c72.wav,speaker_10
61b56abe1bb344de.wav,speaker_08
fb0a3a019a634995.wav,speaker_23
aee66305683d40c0.wav,speaker_04
95ff5eac01a84ed9.wav,speaker_21
82e0b402d50b486e.wav,speaker_10
8437f76ffe4f4b4c.wav,speaker_32
b26d8b048d1e4a02.wav,speaker_19
ebfb4fb3933f445a.wav,speaker_18
c03b314bfc7740f5.wav,speaker_03
0a5a391713464304.wav,speaker_18
1be8e13d3113441a.wav,speaker_05
5ee4e496592e4c81.wav,speaker_06
0aa7f39419e7436e.wav,speaker_03
3655879fec3c47d2.wav,speaker_25
4255c16a6da94fd6.wav,speaker_33
f0d430d3d0864d17.wav,speaker_08
73cda43ab1ab44c6.wav,speaker_31
e68cb5abe1aa434d.wav,speaker_12
3271542fb0314fbf.wav,speaker_25
bfab9c2ac5514813.wav,speaker_04
9b8722616f07477d.wav,speaker_21
772781937c5e45d7.wav,speaker_11
287e94b3b1fc42cb.wav,speaker_28
919c7cfd0db5466e.wav,speaker_21
0bd22e8cd53d4f28.wav,speaker_26
ce5bd8635aaa4f9f.wav,speaker_12
124af97ddcb04ff4.wav,speaker_33
d4aca1e0a3a24dd9.wav,speaker_32
c0339c0973bf4242.wav,speaker_04
3e5adfbdd8aa441a.wav,speaker_18
bf98da481c5a488e.wav,speaker_28
9f885361fb95436d.wav,speaker_28
045a983399e24911.wav,speaker_07
9ee4672c937341b4.wav,speaker_24
7063a4e16aa04ad9.wav,speaker_25
8f5c1e9143b443d1.wav,speaker_17
1c46e72a9f814992.wav,speaker_31
f67ec17b2491475e.wav,speaker_27
14797b3de35c43c6.wav,speaker_20
71772162ee284ec6.wav,speaker_11
85ad637acf5b4991.wav,speaker_06
ca6759ad0998470e.wav,speaker_16
c6c36f54356b4064.wav,speaker_10
030349ad36964873.wav,speaker_35
2fcc8374e6694366.wav,speaker_29
af1ec2b293ca49b8.wav,speaker_33
bea2f589f2ae4877.wav,speaker_33
d9a18a97e57f4dff.wav,speaker_06
b44a9bc97b734f62.wav,speaker_26
0a88cdb6a81a468b.wav,speaker_32
c8e0b6ae0df24dc0.wav,speaker_38
53f49873a4a24c02.wav,speaker_11
b6551cf0ceb9433e.wav,speaker_11
3a4fdc6dc10c4634.wav,speaker_28
a18aa4532d9942c0.wav,speaker_14
107550125e204e0e.wav,speaker_26
a894e3ceacf6460a.wav,speaker_32
f84fff6827ca4914.wav,speaker_13
d681f9c16b024a4a.wav,speaker_39
a074efd2103d43ea.wav,speaker_16
8e986e0c9b214392.wav,speaker_38
84f4a1abc6414cc0.wav,speaker_24
ad25168065e447f7.wav,speaker_02
f4f8f504171a4b30.wav,speaker_39
b36c7783ee9949ac.wav,speaker_01
f2447147a5884ffb.wav,speaker_39
2a4a92cc19984500.wav,speaker_30
bf97a15595eb4404.wav,speaker_32
793ad064ae794edf.wav,speaker_05
0d4b40f2e9e34f95.wav,speaker_23
f4d61689599b42d9.wav,speaker_01
bb6bb7491ac245e3.wav,speaker_23
2840544c41f6496e.wav,speaker_23
1a56432c6e9a43db.wav,speaker_21
27e331a6cbda4f2e.wav,speaker_19
28622b8186ef4d70.wav,speaker_34
4f6e8dd94393463f.wav,speaker_10
3536fc98ae184f37.wav,speaker_21
b9cd258affde4356.wav,speaker_06
2f6faff598ec4816.wav,speaker_37
b9c3bf337fb84df4.wav,speaker_07
b95620a4ca6447ce.wav,speaker_39
c851e445c5ef4e81.wav,speaker_23
c3c405114eef4fcc.wav,speaker_05
75ef2952815f4907.wav,speaker_31
12cd035370854be7.wav,speaker_29
7f0f311f61c349f9.wav,speaker_20
9f54cb4d0b1649ff.wav,speaker_06
bab9bf174516420f.wav,speaker_39
b41628f711b54dd4.wav,speaker_32
753314fb021045fc.wav,speaker_35
b04821774e694a21.wav,speaker_27
488e74cd3607426f.wav,speaker_38
637c33756b934d50.wav,speaker_12
c613b0d6a4d04947.wav,speaker_36
7ffbb15db9fc4095.wav,speaker_39
8481c5f47ce840d3.wav,speaker_13
bc17ed782a064acb.wav,speaker_17
0df17458e77a43d0.wav,speaker_13
61cac7ba71764281.wav,speaker_14
824d1ab481b54c15.wav,speaker_13
15fa9e5926fe4602.wav,speaker_01
eeaba249eceb4464.wav,speaker_16
e9a4db4deae84b36.wav,speaker_06
584e428d37f74f23.wav,speaker_31
f60b5a100ecf4399.wav,speaker_37
c0cbcbd07d3b4686.wav,speaker_27
33457ab6d80e4ca8.wav,speaker_24
5c4283333428488a.wav,speaker_16
bc2a858ca25d42a7.wav,speaker_21
12f4099b4ec542bd.wav,speaker_22
747ec01712db46b6.wav,speaker_39
612cd305062e482a.wav,speaker_26
2a5210f8bfa7451d.wav,speaker_29
f18c02c6a6904b03.wav,speaker_33
23566a6064d74df3.wav,speaker_09
9813f9798f2b4162.wav,speaker_24
a309c97ea5e84ddb.wav,speaker_12
03c8ffbe73734337.wav,speaker_23
5868079c54614bbb.wav,speaker_22
23c3f6c9367a4a92.wav,speaker_22
12e40d6a66954413.wav,speaker_03
0833082cff424e7d.wav,speaker_15
759c0b9ba4bd4831.wav,speaker_37
f90ff013fc514a36.wav,speaker_33
06231c5ce18f4155.wav,speaker_29
913c8ad68ba84efc.wav,speaker_32
8838966ef8c94a25.wav,speaker_14
02686c29c91a4911.wav,speaker_19
b8b922aa5d604070.wav,speaker_01
be61c5f644a64f5d.wav,speaker_02
86ab26faf2f7454f.wav,speaker_34
cf9c07fc2fa64f46.wav,speaker_16
4ce5d3a8fa6a44ed.wav,speaker_15
3ffe0aeb0a464351.wav,speaker_25
0a69647409d14afd.wav,speaker_34
7a580aa53e364cd7.wav,speaker_38
a04ec8ddaf7d486f.wav,speaker_37
f6113826ca5643f7.wav,speaker_17
956b423f61e942b9.wav,speaker_40
9036bca3bb4e4f4c.wav,speaker_38
1b63a16cacd44237.wav,speaker_14
078e5d13f15b480c.wav,speaker_02
feee0c3e1ae94846.wav,speaker_24
8af911265dbf4492.wav,speaker_09
fdd8340c2c19422b.wav,speaker_20
b673ebbe33854cf5.wav,speaker_07
903dffecd3284477.wav,speaker_32
b708d5d6571b4044.wav,speaker_27
7b812c1857374c5a.wav,speaker_28
0fbb44bc018d414d.wav,speaker_16
beab3ce1eefc4d55.wav,speaker_12
8139ffe5167f4b4d.wav,speaker_01
4ab37636903b4508.wav,speaker_10
97c0467f750d4900.wav,speaker_29
db36994523dd4345.wav,speaker_24
6019fa84d83240e1.wav,speaker_37
ff43ffa6a80f4b36.wav,speaker_08
252e24562b4146ac.wav,speaker_39
9fbe548ce7cb41a4.wav,speaker_18
e8b9552e619c40dd.wav,speaker_02
cb255c7a777c40d4.wav,speaker_23
a67f51746edd436d.wav,speaker_25
7601c9e902be4455.wav,speaker_05
3c81b8b7b0de4bab.wav,speaker_02
8bcdafb5b57041e3.wav,speaker_37
9c15e0d9ac9446ad.wav,speaker_06
2ca503e1f8a6475f.wav,speaker_13
302f343d419243c2.wav,speaker_23
4c34ec34d5844330.wav,speaker_08
d06b2ad36630423f.wav,speaker_38
d7d7ab2efbb14d2d.wav,speaker_18
c412cff800714c99.wav,speaker_18
d5de36a213734ef6.wav,speaker_32
86c7c57c24e641a6.wav,speaker_19
ce66620218ba4446.wav,speaker_07
7414856de99c41c4.wav,speaker_39
c91d8d839dd94901.wav,speaker_23
f4ca11e379134e0f.wav,speaker_32
f2dd168272cc4700.wav,speaker_19
005bb92ab942415e.wav,speaker_07
96855a9cc6254c14.wav,speaker_27
71cfa4b486ca447f.wav,speaker_33
502cf87f66194e6d.wav,speaker_26
8d8fe558e63c461a.wav,speaker_27
2946f07f1d344049.wav,speaker_05
4fdb749b92bb4691.wav,speaker_28
634e7af6275d4189.wav,speaker_35
03cceb05dff241f6.wav,speaker_26
fcbc8db66c9d4571.wav,speaker_04
93c80b0e4e7746c9.wav,speaker_09
243c926625344db9.wav,speaker_24
b5b6695309674cc0.wav,speaker_01
1eeda18935b94652.wav,speaker_30
56ce9ab2420f4121.wav,speaker_09
ab07b4373ca04da8.wav,speaker_15
6a5c7e6d79474823.wav,speaker_10
4920e3d62025491b.wav,speaker_20
4dfe692899834358.wav,speaker_12
e946554f090d45c3.wav,speaker_25
135cce723dd7400d.wav,speaker_04
f68c80ac66b04ce3.wav,speaker_02
ce84a76a699e4f6f.wav,speaker_04
f63ad536a7754f9c.wav,speaker_09
dee9bcdc962841f5.wav,speaker_06
3eeb35f52b46404c.wav,speaker_19
f31b8d2a84ee4620.wav,speaker_11
cae0b241b87345cd.wav,speaker_02
2b0f5acdd9954dbd.wav,speaker_39
b407ccd124e04206.wav,speaker_37
256e7000e9e44de5.wav,speaker_07
422a57ea956e48e2.wav,speaker_14
bb76da90026a400d.wav,speaker_18
b56b58b18f0a4e39.wav,speaker_31
cb8042a796ab4250.wav,speaker_29
c9959051211f4123.wav,speaker_22
7d76b9492e4a4986.wav,speaker_38
c61901dc86804f51.wav,speaker_13
2e502038c0e94fac.wav,speaker_21
911b776dfc444483.wav,speaker_03
b5ca23b0ff504eeb.wav,speaker_22
8ced7adbcfa24dde.wav,speaker_25
c919af29bf7d44db.wav,speaker_27
ce1d3e37ee9e4a15.wav,speaker_06
443c68dd82264ae9.wav,speaker_34
730a9a3f64fc441a.wav,speaker_22
d82aa96fd7fa436a.wav,speaker_13
5bd5ad15ec964f8a.wav,speaker_10
dc844ce8a53b41c6.wav,speaker_05
ffa96025f5bb4d4a.wav,speaker_09
d376761bab2241dc.wav,speaker_39
8089957d016b4c5f.wav,speaker_20
32e04a4091cd43f6.wav,speaker_21
559ef3515d214f6a.wav,speaker_25
f4dd1421d16844e4.wav,speaker_17
b5868bfc833c4273.wav,speaker_10
f908944545cb4b91.wav,speaker_15
95cae9ac89cc4f59.wav,speaker_01
9746ead8b9ec4056.wav,speaker_26
66f2247e0b904f7e.wav,speaker_23
b5929d2758dd497b.wav,speaker_13
5b73a129b20a498e.wav,speaker_03
17ea91bd9cbc45e6.wav,speaker_13
4807e349c9364e2b.wav,speaker_32
7cbbb21d3404458a.wav,speaker_25
ae63a342af094d4f.wav,speaker_16
4ff8de9ba25c407d.wav,speaker_26
048ee6efd3884b2a.wav,speaker_17
f77305319c4a4083.wav,speaker_31
aa1a874fbcda4269.wav,speaker_27
e15924d14bb34fc6.wav,speaker_12
562d5416dd8c4730.wav,speaker_06
80b3eca59e544263.wav,speaker_04
ee49bdd534c7477d.wav,speaker_11
cfb2be43c3854fb1.wav,speaker_17
4c2dbd458c174c4e.wav,speaker_35
5b40e5c644e44dc2.wav,speaker_34
60293daf95d84825.wav,speaker_15
d6e4d5254a5a4741.wav,speaker_33
7047bda53a3a49c4.wav,speaker_40
e917c018ca1e4246.wav,speaker_27

