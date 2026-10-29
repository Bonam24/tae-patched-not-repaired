# Benchmark specification

This document specifies the 280-image stress benchmark, the probe sets, and the two
firing criteria of *Patched, Not Repaired: A Low Benign False Positive Rate Certifies
the Region, Not the Model* (TAE @ NeurIPS 2026), so that an independently assembled
set can be compared against ours file by file via the checksums in `checksums/`.

## Families and counts

The frozen stress benchmark is 280 images in six families. None contains damage of
any kind and none was used in training.

| Family             | n  | Role |
|--------------------|----|------|
| Cracked mud        | 50 | deployment-realistic texture |
| Fur                | 40 | instrument (see note below) |
| Gravel             | 50 | deployment-realistic texture |
| Rain on glass      | 50 | deployment-realistic texture |
| Undamaged vehicles | 40 | the one family drawn from the deployment distribution |
| Wall scenes        | 50 | cold control (see note below) |

Additional sets, not part of the frozen benchmark:

| Set                | n  | Role |
|--------------------|----|------|
| Fur probe          | 59 | held out from training, used to select the fur-stage epoch: the 10 images of the 40-image fur family not supplied as negatives, plus 49 collected after training (stored separately; `t`-prefixed copies alongside the fur family are excluded from every benchmark statistic) |
| Noise probe        | 15 | synthetic bandpass stimuli, never supplied in any form; regenerate deterministically with `probes/generate_noise_probe.py` |
| Dry grass          | 52 | post hoc family, collected from a single public web image search after the intervention; never supplied as a negative, never annotated |

Notes. The wall family was gathered as a cold control and turned out to contain palm
fronds, razor wire and loose stone; only 7 of its 50 frames are plain masonry, a post
hoc subset identified after seeing which frames fired, and no separate rate is
reported for them. Of the 40 fur images, 30 were supplied as negatives at the fur
stage of the paper's intervention; the figure that carries the claim is therefore the
59-image probe, not the 40-image family rate.

## Collection protocol

Each benchmark family was chosen before any model was run on it, and the expectation
of whether it would fire was recorded in advance. No image was included or excluded
on the basis of model response. Dry grass is the one exception in timing (collected
after the intervention, chosen because it was expected to be informative) and is
reported as post hoc throughout; its collection also applied no model-response
filtering.

## Preprocessing

Each image is resized to 512x512 for the forward pass (ImageNet mean/std
normalisation), and the probability map is resized back to the native resolution
before any benchmark statistic is computed.

## The two firing criteria

The two rules are not interchangeable and rates under them are never compared.

1. **Operating criterion** (benchmark, dry grass, the paper's Table 3): an image
   *fires* when its maximum predicted damage probability reaches tau = 0.85, the
   operating threshold of the deployed pipeline. *Predicted area* for a benchmark
   family is the fraction of its pixels at or above tau, in the native frame.
2. **Training criterion** (fur and noise probes, the paper's Table 2): an image
   *fires* when the predicted area at threshold 0.5 exceeds 0.001. This is the
   quantity the training loop itself monitored, and areas reported for the probes
   use threshold 0.5.

## Checksums

`checksums/benchmark_sha256.txt` lists `<sha256>  <family>/<filename>` for every
image scored, covering the 280 frozen benchmark images, the 49 later fur images,
the 52 dry grass images and the 15 noise stimuli. Generate or verify with
`make_checksums.py` in this directory.

## Availability

The texture families and fur were gathered from public web image search; we do not
hold their copyright and cannot redistribute the pixels. They are available on
request for non-commercial research use, and the checksums let an independently
assembled set be compared file by file. The 40 undamaged-vehicle images come from
the insurer's corpus, carry registration plates and occasionally people, and are not
available in any form. The noise probe is fully reproducible from the generator
script.
