import './App.css';

import { Howl } from 'howler';
import soundUrl from './assets/demo.mp3';
import React, { useEffect, useRef, useState } from 'react';

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl'; // Import the WebGL backend
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

import { initNotifications, notify } from '@mycv/f8-notification';

var sound = new Howl({
  src: [soundUrl],
});

const NOT_TOUCH_LABEL = 'NOT_TOUCH';
const TOUCHED_LABEL = 'TOUCHED';
const TRAINING_TIMES = 50;
const TOUCHED_CONFIDENCE = 0.8;

function App() {
  const video = useRef();
  const classifier = useRef();
  const canPlaySound = useRef(true);
  const mobilenetModule = useRef();
  const [touched, setTouched] = useState(false);

  const init = async () => {
    await tf.setBackend('webgl'); // Set the backend to WebGL
    await tf.ready(); // Ensure the backend is ready before proceeding

    console.log('Init...');

    await setupCamera();

    console.log('Setup camera success');

    classifier.current = knnClassifier.create();

    mobilenetModule.current = await mobilenet.load(); // đọc ảnh qua DB

    console.log('Setup done');
    console.log("Don't touch your face and press Train 1");

    initNotifications({ cooldown: 3000 });
  };

  const setupCamera = () => {
    // Cách 1
    // return new Promise((resolve, reject) => {
    //   navigator.mediaDevices.getUserMedia =
    //     navigator.mediaDevices.getUserMedia ||
    //     navigator.mediaDevices.webkitGetUserMedia ||
    //     navigator.mediaDevices.mozGetUserMedia ||
    //     navigator.mediaDevices.msGetUserMedia;
    //   if (navigator.mediaDevices.getUserMedia) {
    //     navigator.mediaDevices.getUserMedia(
    //       { video: true },
    //       // xin quyền chạy video -> thành công trả về callBack stream
    //       (stream) => {
    //         video.current.srcObject = stream;
    //         video.current.addEventListener('loadeddata', resolve);
    //       },
    //       (error) => reject(error),
    //     );
    //   } else {
    //     reject();
    //   }
    // });

    // Cách 2
    //   return new Promise((resolve, reject) => {
    //     navigator.mediaDevices
    //       .getUserMedia({ video: true })
    //       .then((stream) => {
    //         video.current.srcObject = stream;
    //         video.current.addEventListener('loadeddata', resolve);
    //       })
    //       .catch((error) => reject(error));
    //   });

    // Cách 3
    return new Promise((resolve, reject) => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then((stream) => {
            video.current.srcObject = stream;
            video.current.addEventListener('loadeddata', () => resolve());
          })
          .catch((error) => {
            console.error('Error accessing the camera: ', error);
            reject(error);
          });
      } else {
        reject(new Error('getUserMedia not supported in this browser.'));
      }
    });
  };

  /**
   * Bước 1: Train cho máy khuôn mặt không chạm tay
   * Bước 2: Train cho máy khuôn mặt có chạm tay
   * Bước 3: Lấy hình ảnh hiện tại, phân tích và so sánh với data đã học trước đó
   * ==> Nếu mà matching với data khuôn mặt chạm tay ==> Cảnh báo
   * @param {*} label
   */

  // Để máy học khuôn mặt của bạn
  const train = async (label) => {
    console.log(`[${label}] Đang train cho máy bắt mặt đẹp trai của bạn...`);

    for (let i = 0; i < TRAINING_TIMES; ++i) {
      console.log(`Progress ${parseInt(((i + 1) / TRAINING_TIMES) * 100)}% training`);

      await training(label);
    }
  };

  // Bắt đầu học khuôn mặt của bạn
  const training = (label) => {
    return new Promise(async (resolve) => {
      const embedding = mobilenetModule.current.infer(video.current, true);

      classifier.current.addExample(embedding, label);
      await sleep(100);
      resolve();
    });
  };

  // Khi ấn nút Run, hàm này chạy và phân tích ảnh hiện tại của bạn
  const run = async () => {
    const embedding = mobilenetModule.current.infer(video.current, true);

    const result = await classifier.current.predictClass(embedding);

    console.log('Label: ', result.label);
    console.log('Confidences: ', result.confidences);

    if (result.label === TOUCHED_LABEL && result.confidences[result.label] > TOUCHED_CONFIDENCE) {
      console.log('Touched');

      if (canPlaySound.current) {
        canPlaySound.current = false;
        sound.play();
        sound.on('end', function () {
          canPlaySound.current = true;
        });
      }

      notify("Don't touch your face", { body: 'You just touched your face!!!' });
      setTouched(true);
    } else {
      console.log('Not touch');
      setTouched(false);
    }

    await sleep(200);

    run();
  };

  const sleep = (ms = 0) => {
    return new Promise((resolve) => setTimeout(resolve, ms));
  };

  useEffect(() => {
    init();

    // cleanup
    return () => {};
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={`main ${touched ? 'touched' : ''}`}>
      <video ref={video} className="video" autoPlay />

      <div className="control">
        <button className="btn" onClick={() => train(NOT_TOUCH_LABEL)}>
          Train 1
        </button>
        <button className="btn" onClick={() => train(TOUCHED_LABEL)}>
          Train 2
        </button>
        <button className="btn" onClick={() => run()}>
          Run
        </button>
      </div>
    </div>
  );
}

export default App;
