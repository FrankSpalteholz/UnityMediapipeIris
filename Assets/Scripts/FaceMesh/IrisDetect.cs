using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    public class IrisDetect : BaseImagePredictor<float>
    {
        public class Result
        {
            //public float score;
            public Vector3[] eyelandmark;
            public Vector3[] irislandmark;
        }
        public const int EYE_KEYPOINT_COUNT = 71;
        public const int IRIS_KEYPOINT_COUNT = 5;
        private float[,] output0 = new float[EYE_KEYPOINT_COUNT, 3]; // eye
        private float[,] output1 = new float[IRIS_KEYPOINT_COUNT, 3]; // iris

        private Result result;
        private Matrix4x4 cropMatrix;

        public Vector2 FaceShift { get; set; } = new Vector2(0f, 0f);
        public Vector2 FaceScale { get; set; } = new Vector2(2f, 2f);
        public Matrix4x4 CropMatrix => cropMatrix;

        public IrisDetect(string modelPath) : base(modelPath, true)
        {
            result = new Result()
            {
                eyelandmark = new Vector3[EYE_KEYPOINT_COUNT],
                irislandmark = new Vector3[IRIS_KEYPOINT_COUNT],
            };
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture inputTex, FaceDetect.Result palm)");
        }

        public void Invoke(Texture inputTex, int[] indxs, FaceDetect.Result face, FaceMesh.Result meshResult, int side)
        {

            CalcEyeRoi(face, meshResult, indxs[0], indxs[1]);

            var options = (inputTex is WebCamTexture)
                ? resizeOptions.GetModifedForWebcam((WebCamTexture)inputTex)
                : resizeOptions;

            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = face.rectEye,
                rotationDegree = CalcFaceRotation(ref face) * Mathf.Rad2Deg,
                shift = FaceShift,
                scale = FaceScale,
                cameraRotationDegree = -options.rotationDegree,
                mirrorHorizontal = options.mirrorHorizontal,
                mirrorVertiacal = options.mirrorVertical,
            });

            RenderTexture rt = resizer.Resize(
                inputTex, options.width, options.height, true,
                cropMatrix,
                TextureResizer.GetTextureST(inputTex, options));

            ToTensor(rt, input0, false);

            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

        }

        private void CalcEyeRoi(FaceDetect.Result face, FaceMesh.Result meshResult, int idx0, int idx1)
        {
            float x0 = meshResult.keypoints[idx0].x;
            float y0 = meshResult.keypoints[idx0].y;
            float x1 = meshResult.keypoints[idx1].x;
            float y1 = meshResult.keypoints[idx1].y;

            float cx = (x0 + x1) / 2.0f;
            float cy = (y0 + y1) / 2.0f;
            float w = Mathf.Abs(x1 - x0);
            float h = Mathf.Abs(y1 - y0);

            {
                float long_side = Mathf.Max(w, h);
                w = h = long_side;
            }
            {
                float scale = 1f;
                w *= scale;
                h *= scale;
            }

            float dx = w / 2.0f;
            float dy = h / 2.0f;

            Vector2[] eye_pos = new Vector2[4];

            eye_pos[0].x = -dx; eye_pos[0].y = -dy;
            eye_pos[1].x = +dx; eye_pos[1].y = -dy;
            eye_pos[2].x = +dx; eye_pos[2].y = +dy;
            eye_pos[3].x = -dx; eye_pos[3].y = +dy;

            float rotation = 0.0f;
            for (int i = 0; i < 4; i++)
            {
                float sx = eye_pos[i].x;
                float sy = eye_pos[i].y;
                eye_pos[i].x = sx * Mathf.Cos(rotation) - sy * Mathf.Sin(rotation);
                eye_pos[i].y = sx * Mathf.Sin(rotation) + sy * Mathf.Cos(rotation);

                eye_pos[i].x += cx;
                eye_pos[i].y += cy;
            }

            face.rectEye.x = cx - w/2;
            face.rectEye.y = (1f - cy) - h/2;
            face.rectEye.width = w;
            face.rectEye.height = h;

        }

        public Result GetResult(int side)
        {
            const float SCALE = 1f / 64f;
            var mtx = cropMatrix.inverse;
            
            
            // output0 = eye
            // output1 = iris

            if(side == -1)
            {
                for (int i = 0; i < EYE_KEYPOINT_COUNT; i++)
                {
                    result.eyelandmark[i] = mtx.MultiplyPoint3x4(new Vector3(
                            1f - output0[i, 0] * SCALE,
                            1f - output0[i, 1] * SCALE,
                            output0[i, 2] * SCALE
                    ));
                }
            }

            if(side == 1)
            {
                for (int i = 0; i < EYE_KEYPOINT_COUNT; i++)
                {
                    result.eyelandmark[i] = mtx.MultiplyPoint3x4(new Vector3(
                            output0[i, 0] * SCALE,
                            1f - output0[i, 1] * SCALE,
                            output0[i, 2] * SCALE
                    ));
                }
            }


            for (int i = 0; i < IRIS_KEYPOINT_COUNT; i++)
            {
                result.irislandmark[i] = mtx.MultiplyPoint3x4(new Vector3(
                        output1[i, 0] * SCALE,
                        1f - output1[i, 1] * SCALE,
                        output1[i, 2] * SCALE
                ));
            }

            return result;
        }

        private static float CalcFaceRotation(ref FaceDetect.Result detection)
        {
            var vec = detection.rightEye - detection.leftEye;
            return -Mathf.Atan2(vec.y, vec.x);
        }
    }
}
