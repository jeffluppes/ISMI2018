x_in = Input(batch_shape=(None, None, None, 3))

### Your code starts here
x = Conv2D(32, (3, 3), activation='relu')(x_in)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(64, (1, 1), activation='relu')(x)
x_out = Conv2D(2, (1, 1), activation='softmax',)(x)

### 
model_2 = Model(x_in, x_out)
model_2.summary()
model_2.compile(optimizer, loss_function)



patch_size = (11,11) # tuple of width, height
stride = 1 # downscaling factor due to pooling

patch_extractor = PatchExtractor(patch_size)
batch_creator = BatchCreator(patch_extractor, train_data)

x, y = batch_creator.create_batch(24)
f, axes = plt.subplots(4, 6)
i = 0;
for ax_row in axes:
    for ax in ax_row:
        ax.imshow(x[i])
        ax.scatter(*[p/2 for p in patch_extractor.patch_size], alpha=0.5)
        i += 1
plt.show()

generator = batch_creator.get_generator(batch_size)

logger_2 = Logger(validation_data, patch_size, stride=stride)

model_2.fit_generator(generator=generator, 
                      steps_per_epoch=steps_per_epoch, 
                      epochs=epochs, 
                      callbacks=[logger_2])