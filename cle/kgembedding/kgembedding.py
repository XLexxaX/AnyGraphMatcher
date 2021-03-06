
#mainly from https://github.com/mana-ysh/knowledge-graph-embeddings
from .kge.optimizer import SGD, Adagrad

def train_model(method='complex', mode='single', dimension=200,
                number_of_epochs=300, batch_size=128,learning_rate=0.05,
                margin=1.0, number_negative_samples=10, optimzer='adagrad',
                l2_regularization=0.0001, gradient_clipping=5, epoch_setp_for_saving=100,
                ratio_complex_dimension=0.5):



    if optimzer == 'sgd':
        opt = SGD(learning_rate)
    elif optimzer == 'adagrad':
        opt = Adagrad(learning_rate)
    else:
        raise NotImplementedError

    if l2_regularization > 0:
        opt.set_l2_reg(l2_regularization)
    if gradient_clipping > 0:
        opt.set_gradclip(gradient_clipping)


    from .kge.transe import TransE
    model = TransE(n_entity=n_entity,
                   n_relation=n_relation,
                   margin=margin,
                   dim=dimension,
                   mode=mode)

    if mode == 'pairwise':
        trainer = PairwiseTrainer(model=model, opt=opt, save_step=args.save_step,
                                  batchsize=args.batch, logger=logger,
                                  evaluator=evaluator, valid_dat=valid_dat,
                                  n_negative=args.negative, epoch=args.epoch,
                                  model_dir=args.log)
    elif mode == 'single':
        trainer = SingleTrainer(model=model, opt=opt, save_step=args.save_step,
                                batchsize=args.batch, logger=logger,
                                evaluator=evaluator, valid_dat=valid_dat,
                                n_negative=args.negative, epoch=args.epoch,
                                model_dir=args.log)

    pass