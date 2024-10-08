import io from 'socket.io-client';

const socket = new io.connect("http://98.153.103.177:51003/");
// const socket = new io.connect("http://localhost:5000/");

// Connect status
socket.on("connect", (msg) => {
  console.log(msg);
});

// Stream Audio
let bufferSize = 2048,
  AudioContext,
  context,
  processor,
  input,
  globalStream;

const mediaConstraints = {
  audio: true,
  video: false
};

let count = 1;

let AudioStreamer = {

  /**
   * @param {object} transcribeConfig Transcription configuration such as language, encoding, etc.
   * @param {function} onData Callback to run on data each time it's received
   * @param {function} onNewJobStarted call back to show start transcribing accepted
   * @param {function} onError Callback to run on an error if one is emitted.
   */
  initRecording: function (transcribeConfig, onData, onNewJobStarted, onError) {
    //socket.emit('startGoogleCloudStream', {...transcribeConfig});
    socket.emit('start');
    AudioContext = window.AudioContext || window.webkitAudioContext;
    context = new AudioContext({
        sampleRate: 16000,
    });
    processor = context.createScriptProcessor(bufferSize, 1, 1);
    processor.connect(context.destination);
    context.resume();

    const handleSuccess = function (stream) {
      globalStream = stream;
      input = context.createMediaStreamSource(stream);
      input.connect(processor);

      processor.onaudioprocess = function (e) {
        microphoneProcess(e);
      };
    };

    navigator.mediaDevices.getUserMedia(mediaConstraints)
      .then(handleSuccess);

    // Socket event from socket server
    // Speech data response
    if (onData) {
      socket.on('speechData', (response) => {
        //console.log('speech data received from server!')
        onData(response.data, response.isMove2NextChunk, response.isPhraseComplete, response.isFinal);
      });
    }

    if (onNewJobStarted) {
      socket.on('onNewJobStarted', (response) => {
        //console.log('start new job status received from server!')
        onNewJobStarted(response.started);
      });
    }

    socket.on('queue_changed', (data) => {
      console.log(`queue_length=${data.queue_len}, last_sample_size=${data.last_sample_size}`);
    })

    socket.on('googleCloudStreamError', (error) => {
      if (onError) {
        onError('error');
      }
      closeAll();
    });

    socket.on('endGoogleCloudStream', () => {
      closeAll();
    });
  },

  stopRecording: function () {
    socket.emit('endGoogleCloudStream');
    socket.emit('stop');
    closeAll();
  }
}

export default AudioStreamer;

// Helper functions
/**
 * Processes microphone data into a data stream
 *
 * @param {object} e Input from the microphone
 */
function microphoneProcess(e) {
  const left = e.inputBuffer.getChannelData(0);
  const left16 = convertFloat32ToInt16(left);
  socket.emit('binaryAudioData', {chunk: left16, count});
  count+=1;
}

/**
 * Converts a buffer from float32 to int16. Necessary for streaming.
 * sampleRateHertz of 16000.
 *
 * @param {object} buffer Buffer being converted
 */
function convertFloat32ToInt16(buffer) {
  let len = buffer.length;
  var output = new Int16Array(len);
  for (var i = 0; i < len; i++){
    var s = Math.max(-1, Math.min(1, buffer[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output.buffer;
}

// include downsampling and transform float32 to int16 ????
// function convertFloat32ToInt16(buffer) {
//   let l = buffer.length;
//   let buf = new Int16Array(l / 3);

//   while (l--) {
//     if (l % 3 === 0) {
//       buf[l / 3] = buffer[l] * 0xFFFF;
//     }
//   }
//   return buf.buffer
// }

/**
 * Stops recording and closes everything down. Runs on error or on stop.
 */
function closeAll() {
  // Clear the listeners (prevents issue if opening and closing repeatedly)
  socket.off('speechData');
  socket.off('googleCloudStreamError');
  let tracks = globalStream ? globalStream.getTracks() : null;
  let track = tracks ? tracks[0] : null;
  if (track) {
    track.stop();
  }

  if (processor) {
    if (input) {
      try {
        input.disconnect(processor);
      } catch (error) {
        console.warn('Attempt to disconnect input failed.')
      }
    }
    processor.disconnect(context.destination);
  }

  if (context) {
    context.close().then(function () {
      input = null;
      processor = null;
      context = null;
      AudioContext = null;
    });
  }
}
