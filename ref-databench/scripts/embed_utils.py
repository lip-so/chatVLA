import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from pathlib import Path
import logging
import json
from typing import List, Union, Optional, Dict, Any, Tuple
import pickle
from io import BytesIO
import base64
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel, BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel  # Using transformers' CLIP instead of clip-by-openai
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VideoLoader:
    """Robust video loader with multiple codec support and fallbacks."""
    
    @staticmethod
    def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """
        Load video frames with robust fallback support for different codecs.
        
        Tries multiple methods:
        1. OpenCV (fastest for supported codecs)
        2. PyAV (good for modern codecs like AV1)
        3. ImageIO with FFmpeg (most robust fallback)
        4. FFmpeg subprocess (ultimate fallback)
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of PIL Images
        """
        video_path = str(video_path)
        
        # Method 1: Try OpenCV first (fastest)
        try:
            frames = VideoLoader._load_with_opencv(video_path, num_frames)
            if frames:
                logger.debug(f"Successfully loaded {len(frames)} frames with OpenCV")
                return frames
        except Exception as e:
            logger.debug(f"OpenCV failed for {video_path}: {e}")
        
        # Method 2: Try PyAV (good for AV1)
        try:
            frames = VideoLoader._load_with_pyav(video_path, num_frames)
            if frames:
                logger.info(f"Successfully loaded {len(frames)} frames with PyAV")
                return frames
        except Exception as e:
            logger.debug(f"PyAV failed for {video_path}: {e}")
        
        # Method 3: Try ImageIO with FFmpeg
        try:
            frames = VideoLoader._load_with_imageio(video_path, num_frames)
            if frames:
                logger.info(f"Successfully loaded {len(frames)} frames with ImageIO")
                return frames
        except Exception as e:
            logger.debug(f"ImageIO failed for {video_path}: {e}")
        
        # Method 4: Ultimate fallback - FFmpeg subprocess
        try:
            frames = VideoLoader._load_with_ffmpeg(video_path, num_frames)
            if frames:
                logger.info(f"Successfully loaded {len(frames)} frames with FFmpeg subprocess")
                return frames
        except Exception as e:
            logger.error(f"FFmpeg subprocess failed for {video_path}: {e}")
        
        # If all methods fail, raise error
        raise ValueError(f"Failed to load video frames from {video_path} using all available methods")
    
    @staticmethod
    def _load_with_opencv(video_path: str, num_frames: int) -> List[Image.Image]:
        """Load video frames using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("OpenCV could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            raise ValueError("No frames detected in video")
        
        # Calculate frame indices
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames could be extracted")
        
        return frames
    
    @staticmethod
    def _load_with_pyav(video_path: str, num_frames: int) -> List[Image.Image]:
        """Load video frames using PyAV (good for AV1 and other modern codecs)."""
        container = av.open(video_path)
        
        if not container.streams.video:
            container.close()
            raise ValueError("No video streams found")
        
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames
        
        # If frame count is unknown, estimate from duration and fps
        if total_frames == 0:
            if video_stream.duration and video_stream.average_rate:
                total_frames = int(video_stream.duration * video_stream.average_rate)
        
        frames = []
        frame_count = 0
        
        # Calculate target frame indices
        if total_frames > 0 and total_frames <= num_frames:
            target_indices = set(range(total_frames))
        elif total_frames > 0:
            target_indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))
        else:
            # Unknown frame count, sample evenly
            target_indices = set(range(0, num_frames * 10, 10))  # Sample every 10th frame
        
        try:
            for frame in container.decode(video_stream):
                if frame_count in target_indices or len(target_indices) == 0:
                    # Convert to PIL Image
                    img_array = frame.to_ndarray(format='rgb24')
                    frames.append(Image.fromarray(img_array))
                    
                    if len(frames) >= num_frames:
                        break
                
                frame_count += 1
                
                # Safety break for unknown frame counts
                if frame_count > num_frames * 100:
                    break
        
        finally:
            container.close()
        
        if not frames:
            raise ValueError("No frames could be extracted with PyAV")
        
        return frames
    
    @staticmethod
    def _load_with_imageio(video_path: str, num_frames: int) -> List[Image.Image]:
        """Load video frames using ImageIO with FFmpeg plugin."""
        # Use imageio with ffmpeg plugin
        reader = imageio.get_reader(video_path, 'ffmpeg')
        
        total_frames = reader.count_frames()
        
        # Calculate frame indices
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            try:
                frame = reader.get_data(idx)
                frames.append(Image.fromarray(frame))
            except Exception as e:
                logger.debug(f"Could not read frame {idx}: {e}")
                continue
        
        reader.close()
        
        if not frames:
            raise ValueError("No frames could be extracted with ImageIO")
        
        return frames
    
    @staticmethod
    def _load_with_ffmpeg(video_path: str, num_frames: int) -> List[Image.Image]:
        """Load video frames using FFmpeg subprocess as ultimate fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First, get video info
            info_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_streams', video_path
            ]
            
            try:
                result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
                import json
                info = json.loads(result.stdout)
                
                # Find video stream
                video_info = None
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_info = stream
                        break
                
                if not video_info:
                    raise ValueError("No video stream found")
                
                # Get duration and frame rate
                duration = float(video_info.get('duration', 0))
                fps = eval(video_info.get('avg_frame_rate', '30/1'))  # Handle fraction format
                total_frames = int(duration * fps) if duration > 0 else num_frames * 2
                
            except Exception as e:
                logger.warning(f"Could not get video info with ffprobe: {e}")
                total_frames = num_frames * 2  # Fallback estimate
            
            # Calculate timestamps for frame extraction
            if total_frames <= num_frames:
                timestamps = [i / fps for i in range(total_frames)] if 'fps' in locals() else [i for i in range(num_frames)]
            else:
                timestamps = np.linspace(0, duration if 'duration' in locals() else num_frames, num_frames)
            
            frames = []
            for i, timestamp in enumerate(timestamps):
                output_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                
                # Extract frame at specific timestamp
                extract_cmd = [
                    'ffmpeg', '-y', '-v', 'quiet',
                    '-ss', str(timestamp),
                    '-i', video_path,
                    '-vframes', '1',
                    '-q:v', '2',  # High quality
                    output_path
                ]
                
                try:
                    subprocess.run(extract_cmd, check=True, capture_output=True)
                    
                    if os.path.exists(output_path):
                        frames.append(Image.open(output_path))
                    
                except subprocess.CalledProcessError as e:
                    logger.debug(f"FFmpeg failed to extract frame at {timestamp}s: {e}")
                    continue
            
            if not frames:
                raise ValueError("No frames could be extracted with FFmpeg")
            
            return frames
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """Get detailed video information for debugging."""
        info = {
            'path': video_path,
            'exists': os.path.exists(video_path),
            'size_mb': os.path.getsize(video_path) / (1024*1024) if os.path.exists(video_path) else 0,
            'opencv_readable': False,
            'pyav_readable': False,
            'ffmpeg_readable': False,
            'codec': 'unknown',
            'resolution': 'unknown',
            'fps': 'unknown',
            'duration': 'unknown'
        }
        
        if not info['exists']:
            return info
        
        # Test OpenCV
        try:
            cap = cv2.VideoCapture(video_path)
            info['opencv_readable'] = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
            cap.release()
        except:
            pass
        
        # Test PyAV
        try:
            container = av.open(video_path)
            if container.streams.video:
                stream = container.streams.video[0]
                info['pyav_readable'] = True
                info['codec'] = stream.codec.name
                info['resolution'] = f"{stream.width}x{stream.height}"
                info['fps'] = float(stream.average_rate) if stream.average_rate else 'unknown'
                info['duration'] = float(stream.duration * stream.time_base) if stream.duration else 'unknown'
            container.close()
        except:
            pass
        
        # Test FFmpeg
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info['ffmpeg_readable'] = True
        except:
            pass
        
        return info

class EmbeddingManager:
    """Manages embedding models and provides unified interface for encoding."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.clip_processor = None
        self.sbert_model = None
        self.blip_model = None
        self.blip_processor = None
        
        # For episode embedding functionality
        self.episode_embedding_cache = {}
        self.prompt_embedding_cache = {}
        
        logger.info(f"Using device: {self.device}")
        
        # Automatically initialize models
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all embedding models with error handling."""
        try:
            self._load_clip()
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
        
        try:
            self._load_sbert()
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT model: {e}")
            self.sbert_model = None
        
        try:
            self._load_blip()
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            self.blip_model = None
            self.blip_processor = None
    
    def _load_clip(self):
        """Load CLIP model using transformers."""
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP model loaded successfully")
    
    def _load_sbert(self):
        """Load Sentence-BERT model."""
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        logger.info("Sentence-BERT model loaded successfully")
    
    def _load_blip(self):
        """Load BLIP model."""
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        logger.info("BLIP model loaded successfully")
    
    def ensure_models_loaded(self):
        """Ensure all models are loaded, reload if necessary."""
        if self.clip_model is None:
            self._load_clip()
        if self.sbert_model is None:
            self._load_sbert()
        if self.blip_model is None or self.blip_processor is None:
            self._load_blip()
    
    def encode_text_clip(self, texts: List[str]) -> torch.Tensor:
        """Encode text using CLIP."""
        if self.clip_model is None or self.clip_processor is None:
            raise RuntimeError("CLIP model not loaded")
        
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features
    
    def encode_text_sbert(self, texts: List[str]) -> np.ndarray:
        if self.sbert_model is None:
            raise RuntimeError("Sentence-BERT model not loaded")
        
        embeddings = self.sbert_model.encode(texts, convert_to_tensor=False)
        return embeddings
    
    def encode_image_clip(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode images using CLIP."""
        if self.clip_model is None or self.clip_processor is None:
            raise RuntimeError("CLIP model not loaded")
        
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def generate_caption_blip(self, image: Image.Image) -> str:
        if self.blip_model is None or self.blip_processor is None:
            raise RuntimeError("BLIP model not loaded")
        
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    
    def load_video_frames(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """
        Load video frames using robust multi-codec video loader.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of PIL Images
        """
        try:
            return VideoLoader.load_video_frames(video_path, num_frames)
        except Exception as e:
            logger.error(f"Failed to load video frames from {video_path}: {e}")
            # Log detailed video info for debugging
            video_info = VideoLoader.get_video_info(video_path)
            logger.error(f"Video info: {video_info}")
            raise
    
    def load_visual_data(self, visual_data) -> List[Image.Image]:
        """Load visual data from various formats (video file, image file, PIL Image, etc.)."""
        if isinstance(visual_data, str):
            # It's a file path
            if visual_data.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return self.load_video_frames(visual_data)
            elif visual_data.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # It's an image file
                return [Image.open(visual_data)]
            else:
                # Try to load as video first, then as image
                try:
                    return self.load_video_frames(visual_data)
                except:
                    try:
                        return [Image.open(visual_data)]
                    except:
                        logger.warning(f"Could not load visual data from path: {visual_data}")
                        return []
        
        elif isinstance(visual_data, dict):
            # Handle different dictionary formats
            if 'type' in visual_data and visual_data['type'] == 'hf_repo_file':
                # It's a HuggingFace repository file reference
                return self._load_hf_repo_video(visual_data)
            elif 'path' in visual_data:
                # It's a path dictionary
                return self.load_visual_data(visual_data['path'])
            elif 'bytes' in visual_data:
                # It's raw bytes data
                from io import BytesIO
                return [Image.open(BytesIO(visual_data['bytes']))]
            else:
                logger.warning(f"Unsupported visual data dictionary format: {visual_data}")
                return []
        
        elif isinstance(visual_data, Image.Image):
            # It's already a PIL Image
            return [visual_data]
        
        elif isinstance(visual_data, list):
            # It's a list of images or paths
            images = []
            for item in visual_data:
                try:
                    if isinstance(item, Image.Image):
                        images.append(item)
                    elif isinstance(item, str):
                        # Try to load as image first, then as video
                        try:
                            images.append(Image.open(item))
                        except:
                            try:
                                video_frames = self.load_video_frames(item)
                                images.extend(video_frames)
                            except:
                                logger.warning(f"Could not load visual data from list item: {item}")
                                continue
                    elif isinstance(item, np.ndarray):
                        # Convert numpy array to PIL Image
                        if item.ndim == 3 and item.shape[2] in [1, 3, 4]:
                            images.append(Image.fromarray(item))
                        else:
                            logger.warning(f"Unsupported numpy array shape: {item.shape}")
                            continue
                    else:
                        logger.warning(f"Unsupported visual data list item type: {type(item)}")
                        continue
                except Exception as e:
                    logger.warning(f"Error processing visual data list item: {e}")
                    continue
            return images
        
        elif isinstance(visual_data, np.ndarray):
            # It's a numpy array (image or video)
            if visual_data.ndim == 3 and visual_data.shape[2] in [1, 3, 4]:
                # Single image
                return [Image.fromarray(visual_data)]
            elif visual_data.ndim == 4:
                # Video (batch of images)
                return [Image.fromarray(frame) for frame in visual_data]
            else:
                logger.warning(f"Unsupported numpy array shape: {visual_data.shape}")
                return []
        
        elif hasattr(visual_data, 'convert'):
            # It's likely a PIL Image or similar
            try:
                return [visual_data.convert('RGB')]
            except:
                logger.warning(f"Could not convert visual data to PIL Image: {type(visual_data)}")
                return []
        
        else:
            logger.warning(f"Unsupported visual data type: {type(visual_data)}")
            return []
    
    def _load_hf_repo_video(self, video_ref: dict) -> List[Image.Image]:
        try:
            from huggingface_hub import hf_hub_download
            
            # Download the video file
            video_path = hf_hub_download(
                repo_id=video_ref['repo_id'],
                filename=video_ref['filename'],
                repo_type='dataset',
                cache_dir=video_ref.get('cache_dir')
            )
            
            # Load frames from the downloaded video
            return self.load_video_frames(video_path)
            
        except Exception as e:
            logger.error(f"Failed to load HF repo video {video_ref['filename']}: {e}")
            raise
    
    def compute_clip_similarity(self, texts: List[str], images: List[Image.Image]) -> float:
        """Compute CLIP similarity between texts and images."""
        text_features = self.encode_text_clip(texts)
        image_features = self.encode_image_clip(images)
        
        # Compute cosine similarity
        similarities = torch.matmul(text_features, image_features.T)
        return similarities.mean().item()
    
    def compute_text_diversity(self, texts: List[str]) -> float:
        """Compute diversity of text embeddings using pairwise cosine similarity."""
        if len(texts) <= 1:
            return 0.0
        
        embeddings = self.encode_text_sbert(texts)
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def compute_visual_diversity(self, images: List[Image.Image]) -> float:
        """Compute visual diversity using CLIP embeddings."""
        if len(images) <= 1:
            return 0.0
        
        image_features = self.encode_image_clip(images)
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(image_features)):
            for j in range(i + 1, len(image_features)):
                sim = torch.dot(image_features[i], image_features[j]).item()
                similarities.append(sim)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def batch_process_videos(self, video_paths: List[str], batch_size: int = 4) -> List[List[Image.Image]]:
        """Process multiple videos in batches."""
        all_frames = []
        
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            batch_frames = []
            
            for path in batch_paths:
                try:
                    frames = self.load_video_frames(path)
                    batch_frames.append(frames)
                except Exception as e:
                    logger.warning(f"Failed to process video {path}: {e}")
                    batch_frames.append([])
            
            all_frames.extend(batch_frames)
        
        return all_frames
    
    def embed_episode(self, episode_data: Dict, num_frames: int = 16) -> np.ndarray:
        """
        Embed a complete robot episode using enhanced video understanding.
        
        Args:
            episode_data: Dictionary containing episode information with 'video' or 'videos' key
            num_frames: Number of frames to sample from the episode video
            
        Returns:
            numpy array representing the episode embedding
        """
        episode_key = str(episode_data.get('episode_index', 'unknown'))
        
        # Check cache first
        if episode_key in self.episode_embedding_cache:
            return self.episode_embedding_cache[episode_key]
        
        try:
            # Extract video data
            video_data = episode_data.get('video')
            if not video_data and 'videos' in episode_data:
                # Use the first available video (prefer 'top' view if available)
                videos = episode_data['videos']
                if isinstance(videos, dict):
                    video_data = videos.get('top', videos.get('front', list(videos.values())[0]))
            
            if not video_data:
                raise ValueError("No video data found in episode")
            
            # Load video frames
            frames = self.load_visual_data(video_data)
            
            # Sample frames evenly if we have more than requested
            if len(frames) > num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Encode frames using CLIP
            if not self.clip_model:
                self._load_clip()
            
            # Convert frames to embeddings
            frame_embeddings = []
            batch_size = 8  # Process in batches to avoid memory issues
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_embeddings = self.encode_image_clip(batch_frames)
                frame_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Concatenate all embeddings
            all_embeddings = np.concatenate(frame_embeddings, axis=0)
            
            # Aggregate frame embeddings into episode embedding
            # Use mean pooling for temporal aggregation
            episode_embedding = np.mean(all_embeddings, axis=0)
            
            # Cache the result
            self.episode_embedding_cache[episode_key] = episode_embedding
            
            logger.info(f"Successfully embedded episode {episode_key} with {len(frames)} frames")
            return episode_embedding
            
        except Exception as e:
            logger.error(f"Failed to embed episode {episode_key}: {e}")
            raise
    
    def embed_prompt(self, prompt: str) -> np.ndarray:
        """
        Embed a text prompt using CLIP text encoder.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array representing the prompt embedding
        """
        # Check cache first
        if prompt in self.prompt_embedding_cache:
            return self.prompt_embedding_cache[prompt]
        
        try:
            if not self.clip_model:
                self._load_clip()
            
            # Encode text using CLIP
            text_embedding = self.encode_text_clip([prompt])
            prompt_embedding = text_embedding.cpu().numpy()[0]  # Get first (and only) embedding
            
            # Cache the result
            self.prompt_embedding_cache[prompt] = prompt_embedding
            
            return prompt_embedding
            
        except Exception as e:
            logger.error(f"Failed to embed prompt '{prompt}': {e}")
            raise
    
    def compare_episode_with_prompt(self, episode_data: Dict, prompt: str, num_frames: int = 16) -> float:
        """
        Compare an episode with a text prompt using cosine similarity.
        
        Args:
            episode_data: Dictionary containing episode information
            prompt: Text prompt to compare with
            num_frames: Number of frames to sample from the episode video
            
        Returns:
            Cosine similarity score between episode and prompt (0-1, higher is more similar)
        """
        try:
            # Get embeddings
            episode_embedding = self.embed_episode(episode_data, num_frames)
            prompt_embedding = self.embed_prompt(prompt)
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                episode_embedding.reshape(1, -1),
                prompt_embedding.reshape(1, -1)
            )[0, 0]
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compare episode with prompt '{prompt}': {e}")
            raise
    
    def find_similar_episodes(self, episodes: List[Dict], query_prompt: str, 
                            top_k: int = 5, num_frames: int = 16) -> List[Tuple[Dict, float]]:
        """
        Find the most similar episodes to a given prompt.
        
        Args:
            episodes: List of episode dictionaries
            query_prompt: Text prompt to search for
            top_k: Number of top similar episodes to return
            num_frames: Number of frames to sample from each episode
            
        Returns:
            List of (episode, similarity_score) tuples, sorted by similarity (highest first)
        """
        try:
            episode_similarities = []
            
            logger.info(f"Comparing {len(episodes)} episodes with prompt: '{query_prompt}'")
            
            for i, episode in enumerate(episodes):
                try:
                    similarity = self.compare_episode_with_prompt(episode, query_prompt, num_frames)
                    episode_similarities.append((episode, similarity))
                    
                    if i % 10 == 0:  # Log progress every 10 episodes
                        logger.info(f"Processed {i + 1}/{len(episodes)} episodes")
                        
                except Exception as e:
                    logger.warning(f"Failed to process episode {i}: {e}")
                    continue
            
            # Sort by similarity (highest first)
            episode_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            results = episode_similarities[:top_k]
            
            logger.info(f"Found {len(results)} similar episodes. Top similarity: {results[0][1]:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar episodes: {e}")
            raise
    
    def batch_embed_episodes(self, episodes: List[Dict], num_frames: int = 16) -> np.ndarray:
        """
        Embed multiple episodes in batch for efficiency.
        
        Args:
            episodes: List of episode dictionaries
            num_frames: Number of frames to sample from each episode
            
        Returns:
            numpy array of shape (num_episodes, embedding_dim) containing all episode embeddings
        """
        try:
            embeddings = []
            
            logger.info(f"Batch embedding {len(episodes)} episodes...")
            
            for i, episode in enumerate(episodes):
                try:
                    embedding = self.embed_episode(episode, num_frames)
                    embeddings.append(embedding)
                    
                    if i % 10 == 0:
                        logger.info(f"Embedded {i + 1}/{len(episodes)} episodes")
                        
                except Exception as e:
                    logger.warning(f"Failed to embed episode {i}: {e}")
                    # Add zero embedding as placeholder
                    if embeddings:
                        embeddings.append(np.zeros_like(embeddings[0]))
                    else:
                        # If this is the first episode and it fails, we need to determine embedding size
                        # Use a dummy embedding of standard CLIP size
                        embeddings.append(np.zeros(768))  # ViT-L/14 embedding size
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to batch embed episodes: {e}")
            raise
    
    def clear_cache(self):
        """Clear embedding caches to free memory."""
        self.episode_embedding_cache.clear()
        self.prompt_embedding_cache.clear()
        logger.info("Embedding caches cleared")