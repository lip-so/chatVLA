document.addEventListener('DOMContentLoaded', function() {
  const typingElement = document.querySelector('.landing-title');
  if (!typingElement) return;

  const phrases = [
    "Tune your robot. Instantly.",
    "Control it with conversation.",
    "Fine-tune policies, not patience.",
    "Deploy from the cloud to the edge."
  ];

  let phraseIndex = 0;
  let charIndex = 0;
  let isDeleting = false;

  function type() {
    const currentPhrase = phrases[phraseIndex];
    const speed = isDeleting ? 75 : 150;

    typingElement.textContent = currentPhrase.substring(0, charIndex);

    if (!isDeleting && charIndex < currentPhrase.length) {
      charIndex++;
    } else if (isDeleting && charIndex > 0) {
      charIndex--;
    } else {
      isDeleting = !isDeleting;
      if (!isDeleting) {
        phraseIndex = (phraseIndex + 1) % phrases.length;
      }
    }

    const pauseTime = (isDeleting || charIndex === currentPhrase.length) 
        ? (isDeleting ? speed : 2000)
        : speed;

    setTimeout(type, pauseTime);
  }

  type();
}); 