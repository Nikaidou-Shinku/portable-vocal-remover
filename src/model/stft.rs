use ndarray::{Array1, Array2, Array3, Array4, Axis, Slice};
use realfft::RealFftPlanner;

fn hann_window_periodic(window_length: usize) -> Array1<f32> {
  if window_length == 0 {
    return Array1::zeros(0);
  }

  if window_length == 1 {
    return Array1::ones(1);
  }

  let half_length = window_length / 2 + 1;
  let scaling = (std::f32::consts::PI * 2.0) / window_length as f32;

  let mut res = Array1::zeros(window_length);

  for i in 1..half_length {
    let cur = 0.5 - 0.5 * (scaling * i as f32).cos();
    res[i] = cur;
    res[window_length - i] = cur;
  }

  res
}

// window = hann_window
// center = True
// pad_mode = 'reflect'
// onesided = True
// return_complex = False
fn stft(input: Array2<f32>, n_fft: usize, hop_length: usize) -> Array4<f32> {
  let (batch_num, length) = input.dim();
  let freq_num = n_fft / 2 + 1;
  let frame_num = length / hop_length + 1;

  let window = hann_window_periodic(n_fft);

  let mut planner = RealFftPlanner::<f32>::new();
  let fft = planner.plan_fft_forward(n_fft);
  let mut scratch = fft.make_scratch_vec();

  // NOTE: the shape is different from `torch.stft`!
  let mut res = Array4::zeros((batch_num, 2, freq_num, frame_num));

  for batch in 0..batch_num {
    // TODO(perf): try to use slicing instead of indexing
    let at = |pos: isize| {
      if pos < 0 {
        let pos: usize = (-pos).try_into().unwrap();
        return input[[batch, pos]];
      }

      // this should always success
      let pos: usize = pos.try_into().unwrap();

      if pos >= length {
        let pos = length * 2 - pos - 2;
        return input[[batch, pos]];
      }

      input[[batch, pos]]
    };

    for (frame_id, frame_center) in (0..=length).step_by(hop_length).enumerate() {
      let left_num = n_fft / 2;
      let right_num = n_fft - left_num;

      let left_num: isize = left_num.try_into().unwrap();
      let right_num: isize = right_num.try_into().unwrap();

      let frame_center: isize = frame_center.try_into().unwrap();

      let mut frame: Vec<f32> = ((frame_center - left_num)..(frame_center + right_num))
        .map(at)
        .zip(&window)
        .map(|(a, b)| a * b)
        .collect();

      let mut cur = fft.make_output_vec();

      fft
        .process_with_scratch(&mut frame, &mut cur, &mut scratch)
        .expect("Failed to process fft");
      drop(frame);

      assert_eq!(cur.len(), freq_num);

      // TODO(perf): maybe do some optimization
      for i in 0..freq_num {
        res[[batch, 0, i, frame_id]] = cur[i].re;
        res[[batch, 1, i, frame_id]] = cur[i].im;
      }
    }
  }

  res
}

fn istft() {
  todo!()
}

pub struct Stft {
  n_fft: usize,
  hop_length: usize,
  dim_f: usize,
}

impl Stft {
  pub fn new(n_fft: usize, hop_length: usize, dim_f: usize) -> Self {
    Self {
      n_fft,
      hop_length,
      dim_f,
    }
  }

  pub fn apply(&self, x: Array3<f32>) -> Array4<f32> {
    let (b, c, t) = x.dim();
    let x = x.into_shape((b * c, t)).unwrap();
    let x = stft(x, self.n_fft, self.hop_length);
    let (_, _, _, frame_num) = x.dim();
    let rem = x.len() / frame_num / b / c / 2;
    let mut x = x.into_shape((b, c * 2, rem, frame_num)).unwrap();
    if rem > self.dim_f {
      x.slice_axis_inplace(Axis(2), Slice::from(0..self.dim_f));
    }
    x
  }

  pub fn inverse(&self, x: Array4<f32>) -> Array3<f32> {
    let (b, c, f, t) = x.dim();
    todo!("istft")
  }
}

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use super::{hann_window_periodic, Stft};

  #[test]
  fn hann() {
    let window = hann_window_periodic(16);
    let expected = arr1(&[
      0.0,
      0.038060248,
      0.14644662,
      0.3086583,
      0.5,
      0.69134176,
      0.8535534,
      0.9619398,
      1.0,
      0.9619398,
      0.8535534,
      0.69134176,
      0.5,
      0.3086583,
      0.14644662,
      0.038060248,
    ]);

    assert_eq!(window, expected);

    let window = hann_window_periodic(23);
    let expected = arr1(&[
      0.0,
      0.018541366,
      0.072790295,
      0.15872344,
      0.2699675,
      0.39827198,
      0.5341212,
      0.6674398,
      0.7883402,
      0.88785565,
      0.95860565,
      0.99534297,
      0.99534297,
      0.95860565,
      0.88785565,
      0.7883402,
      0.6674398,
      0.5341212,
      0.39827198,
      0.2699675,
      0.15872344,
      0.072790295,
      0.018541366,
    ]);

    assert_eq!(window, expected);
  }

  #[test]
  fn stft() {
    let input = arr1(&[
      0.7694, 1.1859, 0.5551, 0.1606, -2.7012, 1.9168, -1.1126, -0.7238, 0.5741, 0.3924, -0.4585,
      -0.1567, 0.3945, -1.1792, -0.9736, -0.0393, 0.7241, -0.4223, 1.2212, -0.2810, 0.0122, 0.3384,
      1.2478, -0.5534, 0.0191, -1.1349, -1.5956, 1.3454, 1.6886, 0.7180,
    ])
    .into_shape((1, 1, 30))
    .unwrap();

    let stft = Stft::new(20, 5, 3072);
    let res = stft.apply(input);

    let expected = arr1(&[
      1.7849245,
      0.13593864,
      -1.8351681,
      -0.47893012,
      0.78542936,
      2.0766218,
      2.59187,
      -3.1455154,
      1.2482634,
      0.7564946,
      0.95189947,
      -1.6473615,
      0.39937252,
      -3.2425263,
      4.5801783,
      -2.9697003,
      1.2706327,
      -1.0049411,
      1.9606514,
      -4.5927815,
      4.668249,
      -3.5540538,
      1.304317,
      -1.3828998,
      0.18344367,
      0.93397415,
      4.869222,
      -3.8699167,
      1.2108731,
      0.9310956,
      0.3449924,
      -0.42736113,
      -2.0460293,
      -2.7096195,
      1.0185671,
      2.892271,
      -3.0063825,
      1.8473498,
      -0.74834496,
      0.64804554,
      1.6762886,
      -0.25526416,
      -4.1347394,
      4.7423406,
      -4.220063,
      2.8689265,
      -2.1128309,
      0.1759075,
      1.6606522,
      -1.9091977,
      -4.2965345,
      1.632443,
      -1.7967945,
      2.5221887,
      -1.5410514,
      -1.9074152,
      4.6121645,
      4.1815605,
      1.773217,
      0.0107010305,
      0.21606487,
      0.96005857,
      -0.20337749,
      1.8694959,
      -4.8336635,
      -0.5608875,
      1.6062963,
      -2.5178468,
      0.27066815,
      0.83212256,
      -6.6278763,
      5.261469,
      -1.087389,
      -2.8087206,
      3.3008585,
      -1.092751,
      0.0059478283,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      -0.000000059604645,
      -1.6170566,
      -0.39044082,
      1.7158488,
      -0.574148,
      1.8997475,
      -1.3757086,
      0.000000019092113,
      -0.26014453,
      1.3804486,
      -2.048101,
      0.4908269,
      -1.4831316,
      4.020174,
      0.000000029802322,
      1.591782,
      -0.4762711,
      2.0980077,
      1.7205341,
      -1.1053863,
      -4.167661,
      -0.00000011920929,
      -1.063128,
      -0.5726812,
      -1.6970809,
      -1.0973073,
      0.6810014,
      0.4161471,
      -0.0,
      0.99606836,
      -1.0679314,
      1.1240335,
      -0.3384118,
      -0.5236225,
      2.672924,
      -0.00000011920929,
      -1.7475617,
      1.2395093,
      -0.79212713,
      -0.74536157,
      3.3762443,
      -4.140044,
      0.000000029802322,
      2.2008955,
      1.2098734,
      0.6472564,
      0.6446476,
      -3.2901611,
      3.4485805,
      0.000000019092113,
      -1.6036186,
      -0.3337717,
      -0.8505293,
      0.011720717,
      1.5446546,
      -0.66244113,
      -0.000000059604645,
      1.3439155,
      -2.6982756,
      1.0821322,
      0.5387414,
      -1.5806721,
      -0.56254596,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ])
    .into_shape((1, 2, 11, 7))
    .unwrap();

    assert_eq!(res, expected);
  }
}
