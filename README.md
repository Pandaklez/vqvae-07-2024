# vqvae-07-2024

-----------

**TL;DR**: Random codebook restarts, reading dataset from zip, pose normalization

-----------

`VQVAE_norm_by_meanpose-nowarmup.ipynb` notebook has an okay example of model trained on SSLC corpus data, that was wrapped up into zip file for speed reasons. The data was normalized based on mean pose data (basically with z-score, based on Amit's idea). Zip file wrapping is also based on Amit's code.

Best model to the day (05-07-2024) specs:

* `vq_vae_restart-nowarmup-from-zip-noamp-200epochs.pth` - the video for this model was trimmed from 110 beginning frames and 40 end frames, commitment beta is going from 0.35(actually less-0.26) to 0.001 with smoothing 0.1, no warm up, codebook random restarts for dead codes each epoch, Xavier initialization for embeddings. The data is normalized with `(data - mean)/std` with the help of mean pose. Test data is also normalized. Dataset is wrapped in zip. No mixed precision amp with float16/half. Trained on 200 epochs. The training end perplexity is **219.824**. - reconstuction loss and total loss are `tensor(0.6316, device='cuda:0') tensor(2.0421, device='cuda:0')`.
* As you can see total loss is much a higher than reconstruction loss, before introducing zipped dataset they were more comparable. The difference comes from a spike in codebook/commitment loss. I don't understand yet why. Here are the graphs from `wandb`:

[!training_charts](./example-reconstructions/vqvae-restartcodebook-zipdata.png)

In addition I tried to employ mixed precision training but for some reason it was dropping the scores and reconstruction quality.
