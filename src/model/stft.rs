// TODO: remove `unwrap`
use ndarray::{
  concatenate, s, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, Axis, Slice,
  Zip,
};
use realfft::{num_complex::Complex, RealFftPlanner};

fn hann_window_periodic(window_length: usize) -> Array1<f64> {
  if window_length == 0 {
    return Array1::zeros(0);
  }

  if window_length == 1 {
    return Array1::ones(1);
  }

  let half_length = window_length / 2 + 1;
  let scaling = (std::f64::consts::PI * 2.0) / window_length as f64;

  let mut res = Array1::zeros(window_length);

  for i in 1..half_length {
    let cur = 0.5 - 0.5 * (scaling * i as f64).cos();
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
fn stft(input: ArrayView2<f64>, n_fft: usize, hop_length: usize) -> Array4<f64> {
  let (batch_num, length) = input.dim();
  let freq_num = n_fft / 2 + 1;
  let frame_num = length / hop_length + 1;

  let window = hann_window_periodic(n_fft);

  let mut planner = RealFftPlanner::<f64>::new();
  let fft = planner.plan_fft_forward(n_fft);
  let mut scratch = fft.make_scratch_vec();

  let left_num = n_fft / 2;
  let right_num = n_fft - left_num;
  let left_num: isize = left_num.try_into().unwrap();
  let right_num: isize = right_num.try_into().unwrap();

  // NOTE: the shape is different from `torch.stft`!
  let mut res = Array4::zeros((batch_num, 2, freq_num, frame_num));

  for batch in 0..batch_num {
    for (frame_id, frame_center) in (0..=length).step_by(hop_length).enumerate() {
      let frame_center: isize = frame_center.try_into().unwrap();

      let left = frame_center - left_num;
      let raw_right = frame_center + right_num;
      let right: usize = raw_right.try_into().unwrap();

      let mut frame = if left >= 0 && right <= length {
        input.slice(s![batch, left..raw_right]).to_owned()
      } else {
        let mut parts = Vec::new();

        let left: usize = if left < 0 {
          parts.push(input.slice(s![batch, 1..(1 - left);-1]));
          0
        } else {
          left.try_into().unwrap()
        };

        if right > length {
          parts.push(input.slice(s![batch, left..length]));
          parts.push(input.slice(s![batch, (length * 2 - right - 1)..(length - 1);-1]));
        } else {
          parts.push(input.slice(s![batch, left..right]));
        }

        concatenate(Axis(0), &parts).unwrap()
      };

      Zip::from(&mut frame).and(&window).for_each(|a, b| *a *= b);
      let mut frame: Vec<_> = frame.into_iter().collect();

      let mut cur = fft.make_output_vec();
      assert_eq!(cur.len(), freq_num);

      fft
        .process_with_scratch(&mut frame, &mut cur, &mut scratch)
        .expect("Failed to process fft");

      // TODO(perf): maybe do some optimization
      for i in 0..freq_num {
        res[[batch, 0, i, frame_id]] = cur[i].re;
        res[[batch, 1, i, frame_id]] = cur[i].im;
      }
    }
  }

  res
}

fn istft(input: ArrayView4<f64>, n_fft: usize, hop_length: usize) -> Array2<f64> {
  let (batch_num, _, freq_num, frame_num) = input.dim();
  // this may shorter than original length
  let length = (frame_num - 1) * hop_length;

  let window = hann_window_periodic(n_fft);

  let mut planner = RealFftPlanner::<f64>::new();
  let fft = planner.plan_fft_inverse(n_fft);
  let mut scratch = fft.make_scratch_vec();

  let left_num = n_fft / 2;

  let mut res = Array2::zeros((batch_num, length));
  let mut divider = Array2::<f64>::zeros((batch_num, length));

  for batch in 0..batch_num {
    for frame_id in 0..frame_num {
      let mut cur = Vec::with_capacity(freq_num);

      for i in 0..freq_num {
        cur.push(Complex::new(
          input[[batch, 0, i, frame_id]],
          input[[batch, 1, i, frame_id]],
        ));
      }

      // TODO: check this
      cur[0].im = 0.0;
      cur[freq_num - 1].im = 0.0;

      let mut frame = fft.make_output_vec();

      fft
        .process_with_scratch(&mut cur, &mut frame, &mut scratch)
        .expect("Failed to process inverse fft");

      let frame_center = frame_id * hop_length;

      let left = if frame_center < left_num {
        left_num - frame_center
      } else {
        0
      };

      let right = if frame_center + n_fft >= length + left_num + 1 {
        length + left_num - frame_center
      } else {
        n_fft
      };

      for i in left..right {
        let pos = frame_center + i - left_num;
        res[[batch, pos]] += frame[i] * window[i] / n_fft as f64;
        divider[[batch, pos]] += window[i] * window[i];
      }
    }
  }

  res / divider
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

  pub fn apply(&self, x: ArrayView3<f64>) -> Array4<f64> {
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

  pub fn inverse(&self, x: ArrayView4<f64>) -> Array3<f64> {
    let (b, c, f, t) = x.dim();
    let n = self.n_fft / 2 + 1;

    let res = if f < n {
      let f_pad = Array4::zeros((b, c, n - f, t));
      let x = concatenate(Axis(2), &[x.view(), f_pad.view()]).unwrap();
      let rem = b * c / 2;
      let x = Array4::from_shape_vec((rem, 2, n, t), x.into_iter().collect()).unwrap();
      istft(x.view(), self.n_fft, self.hop_length)
    } else {
      let rem = b * c / 2;
      let x = x.into_shape((rem, 2, n, t)).unwrap();
      istft(x, self.n_fft, self.hop_length)
    };

    let rem = res.len() / b / 2;
    res.into_shape((b, 2, rem)).unwrap()
  }
}
