import tensorflow as tf


def train_step(args, model, optimizer, train_ds, epoch, loss_object, train_summary_writer, train_loss, train_accuracy):
    for step, (images, labels) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(labels, predictions)
        if step % args.log_interval == 0:
            print("Epoch: {} \t\ntrain_loss={:.4f}\ntrain_accuracy={:.4f}\n".format(
                epoch + 1,
                train_loss.result(), train_accuracy.result() * 100)
            )
    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
        tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)


def test_step(model, test_ds, epoch, loss_object, test_summary_writer, test_loss, test_accuracy):
    for (images, labels) in test_ds:
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    with test_summary_writer.as_default():
        tf.summary.scalar('val_loss', test_loss.result(), step=epoch)
        tf.summary.scalar('val_accuracy', test_accuracy.result(), step=epoch)
    print("val_loss={:.4f}\nval_accuracy={:.4f}\n".format(
        test_loss.result(), test_accuracy.result() * 100)
    )