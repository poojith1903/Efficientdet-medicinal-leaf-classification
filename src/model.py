# === STEP 3: BiFPN ===
def bifpn_block(features, filters):
    p3, p4, p5 = features
    p3 = Conv2D(filters, 1, padding='same')(p3)
    p4 = Conv2D(filters, 1, padding='same')(p4)
    p5 = Conv2D(filters, 1, padding='same')(p5)
    p5_up = UpSampling2D()(p5)
    p4_td = Add()([p4, p5_up])
    p4_td = Conv2D(filters, 3, padding='same', activation='relu')(p4_td)
    p4_up = UpSampling2D()(p4_td)
    p3_td = Add()([p3, p4_up])
    p3_td = Conv2D(filters, 3, padding='same', activation='relu')(p3_td)
    return p3_td, p4_td, p5

# === STEP 4: Model ===
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(512,512, 3))
base_model.trainable = False

inputs = Input(shape=(512,512, 3))
p3 = base_model.get_layer('block4a_expand_activation').output
p4 = base_model.get_layer('block6a_expand_activation').output
p5 = base_model.get_layer('top_activation').output

p3, p4, p5 = bifpn_block([p3, p4, p5], filters=128)
p3_pool = GlobalAveragePooling2D()(p3)
p4_pool = GlobalAveragePooling2D()(p4)
p5_pool = GlobalAveragePooling2D()(p5)

x = Add()([p3_pool, p4_pool, p5_pool])
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# === STEP 5: Compile and Train ===
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(r"C:\Users\LENOVO\Downloads\capstone project\efficientnet_bifpn_80class.keras", save_best_only=True)
earlystop = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint, earlystop]
)

model.save(r"C:\Users\LENOVO\Downloads\capstone project\efficientnet_bifpn_80class_final.keras")
