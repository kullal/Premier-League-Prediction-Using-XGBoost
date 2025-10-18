import React, { useRef, useEffect } from "react";
import { gsap } from "gsap";

const calculateSpotlightValues = radius => ({
  proximity: radius * 0.5,
  fadeDistance: radius * 0.75
});

const updateCardGlowProperties = (card, mouseX, mouseY, glow, radius) => {
  const rect = card.getBoundingClientRect();
  const relativeX = ((mouseX - rect.left) / rect.width) * 100;
  const relativeY = ((mouseY - rect.top) / rect.height) * 100;

  card.style.setProperty('--glow-x', `${relativeX}%`);
  card.style.setProperty('--glow-y', `${relativeY}%`);
  card.style.setProperty('--glow-intensity', glow.toString());
  card.style.setProperty('--glow-radius', `${radius}px`);
};

function GlobalSpotlight({ gridRef, enabled = true, spotlightRadius = 300, glowColor = '34, 104, 57' }) {
  useEffect(() => {
    if (!enabled || !gridRef?.current) return;

    const spotlight = document.createElement('div');
    spotlight.className = 'global-spotlight';
    spotlight.style.cssText = `
      position: fixed;
      width: ${spotlightRadius * 2}px;
      height: ${spotlightRadius * 2}px;
      border-radius: 50%;
      pointer-events: none;
      background: radial-gradient(circle,
        rgba(${glowColor}, 0.14) 0%,
        rgba(${glowColor}, 0.06) 20%,
        transparent 60%);
      z-index: 200;
      opacity: 0;
      transform: translate(-50%, -50%);
      mix-blend-mode: screen;
      transition: opacity 0.15s linear;
    `;
    document.body.appendChild(spotlight);

    const handleMouseMove = e => {
      const section = gridRef.current;
      if (!section) return;
      const rect = section.getBoundingClientRect();
      const inside =
        e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;

      const cards = section.querySelectorAll('.card');

      if (!inside) {
        gsap.to(spotlight, { opacity: 0, duration: 0.25, ease: 'power2.out' });
        cards.forEach(c => c.style.setProperty('--glow-intensity', '0'));
        return;
      }

      const { proximity, fadeDistance } = calculateSpotlightValues(spotlightRadius);
      let minDistance = Infinity;

      cards.forEach(card => {
        const r = card.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const distance = Math.hypot(e.clientX - cx, e.clientY - cy) - Math.max(r.width, r.height) / 2;
        const effectiveDistance = Math.max(0, distance);
        minDistance = Math.min(minDistance, effectiveDistance);

        let glowIntensity = 0;
        if (effectiveDistance <= proximity) glowIntensity = 1;
        else if (effectiveDistance <= fadeDistance)
          glowIntensity = (fadeDistance - effectiveDistance) / (fadeDistance - proximity);

        updateCardGlowProperties(card, e.clientX, e.clientY, glowIntensity, spotlightRadius);
      });

      gsap.to(spotlight, { left: e.clientX, top: e.clientY, duration: 0.08, ease: 'power2.out' });

      const targetOpacity =
        minDistance <= proximity ? 0.6 : minDistance <= fadeDistance ? ((fadeDistance - minDistance) / (fadeDistance - proximity)) * 0.6 : 0;

      gsap.to(spotlight, { opacity: targetOpacity, duration: 0.12, ease: 'power2.out' });
    };

    const handleLeave = () => {
      gridRef.current?.querySelectorAll('.card').forEach(c => c.style.setProperty('--glow-intensity', '0'));
      gsap.to(spotlight, { opacity: 0, duration: 0.25, ease: 'power2.out' });
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleLeave);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleLeave);
      spotlight.remove();
    };
  }, [gridRef, enabled, spotlightRadius, glowColor]);

  return null;
}

export default function About() {
  const contentRef = useRef(null);

  return (
    <div className="mt-36 relative min-h-screen bg-black text-white overflow-visible">
      <style>
        {`
          .about-section {
            --glow-x: 50%;
            --glow-y: 50%;
            --glow-intensity: 0;
            --glow-radius: 200px;
            --glow-color: 34, 104, 57;
          }

          .card {
            position: relative;
            transition: box-shadow .25s ease, transform .2s ease;
          }

          .card::after {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at var(--glow-x) var(--glow-y),
              rgba(var(--glow-color), calc(var(--glow-intensity) * 0.9)) 0%,
              rgba(var(--glow-color), calc(var(--glow-intensity) * 0.4)) 25%,
              transparent 55%);
            border-radius: inherit;
            pointer-events: none;
            z-index: 0;
            mix-blend-mode: screen;
            opacity: calc(var(--glow-intensity));
            transition: opacity .18s linear;
          }
        `}
      </style>
      <div ref={contentRef} className="about-section relative z-10 flex flex-col items-center justify-center min-h-screen p-8">
        <GlobalSpotlight gridRef={contentRef} enabled={true} spotlightRadius={300} glowColor={'34, 104, 57'} />
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-bold mb-8 drop-shadow-lg">
            About EPL Match Predictor
          </h1>
          <p className="text-lg md:text-xl text-gray-300 mb-12 drop-shadow-md leading-relaxed">
            This English Premier League match prediction application is designed
            to help you analyze and predict match outcomes based on advanced
            XGBoost models.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            <div className="card bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6" style={{ '--glow-x': '50%', '--glow-y': '50%', '--glow-intensity': 0, position: 'relative' }}>
              <div style={{ position: 'relative', zIndex: 1 }}>
                <h3 className="text-2xl font-bold mb-4">Project Overview</h3>
                <p className="text-gray-300 leading-relaxed">
                  The Premier League Prediction project uses XGBoost machine
                  learning algorithm to predict English Premier League match
                  outcomes with high accuracy. We analyze historical match data,
                  team statistics, player performance, and various other factors
                  to provide reliable predictions.
                </p>
                <p className="text-gray-300 leading-relaxed mt-4">
                  This application is the result of in-depth research into factors
                  influencing football match outcomes, with special focus on the
                  English Premier League. Using data from recent seasons, we have
                  trained a model capable of understanding patterns and trends in
                  football matches.
                </p>
              </div>
            </div>

            <div className="card bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6" style={{ '--glow-x': '50%', '--glow-y': '50%', '--glow-intensity': 0, position: 'relative' }}>
              <div style={{ position: 'relative', zIndex: 1 }}>
                <h3 className="text-2xl font-bold mb-4">Key Features</h3>
                <ul className="text-gray-300 text-left list-disc list-inside space-y-2">
                  <li>
                    <strong>Match Outcome Prediction:</strong> Predict upcoming
                    season match results with high accuracy.
                  </li>
                  <li>
                    <strong>Historical Match Analysis:</strong> In-depth analysis
                    of previous matches between two teams, helping users
                    understand historical patterns and trends.
                  </li>
                  <li>
                    <strong>Advanced XGBoost Model:</strong> Uses state-of-the-art
                    XGBoost algorithm for more accurate predictions compared to
                    traditional statistical methods.
                  </li>
                  <li>
                    <strong>User-Friendly Interface:</strong> Intuitive design
                    allowing users to quickly obtain predictions and analysis.
                  </li>
                  <li>
                    <strong>Deep Analysis:</strong> Beyond predictions, the app
                    provides in-depth analysis of factors influencing match
                    outcomes.
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            <div className="card bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6" style={{ '--glow-x': '50%', '--glow-y': '50%', '--glow-intensity': 0, position: 'relative' }}>
              <div style={{ position: 'relative', zIndex: 1 }}>
                <h3 className="text-2xl font-bold mb-4">Technical Details</h3>
                <p className="text-gray-300 leading-relaxed mb-4">
                  <strong>Technologies Used:</strong>
                </p>
                <ul className="text-gray-300 text-left list-disc list-inside space-y-1">
                  <li>Python as the primary programming language</li>
                  <li>XGBoost for machine learning algorithm</li>
                  <li>Pandas and NumPy for data manipulation</li>
                  <li>React.js for web interface development</li>
                  <li>Flask for backend API</li>
                </ul>
                <p className="text-gray-300 leading-relaxed mt-4">
                  <strong>Data Sources:</strong> Data used in this project comes
                  from trusted sources providing English Premier League
                  statistics, including match results, team and player statistics,
                  and various performance metrics.
                </p>
              </div>
            </div>

            <div className="card bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6" style={{ '--glow-x': '50%', '--glow-y': '50%', '--glow-intensity': 0, position: 'relative' }}>
              <div style={{ position: 'relative', zIndex: 1 }}>
                <h3 className="text-2xl font-bold mb-4">Development Team</h3>
                <div className="text-gray-300 text-left space-y-4">
                  <div>
                    <p className="font-semibold">Riski Yuniar Pratama</p>
                    <p className="text-sm">Model Builder</p>
                  </div>
                  <div>
                    <p className="font-semibold">Gangsar Reka Pambudi</p>
                    <p className="text-sm">
                      Project Manager & Frontend Developer
                    </p>
                  </div>
                  <div>
                    <p className="font-semibold">Yumna Salma Salsabilla</p>
                    <p className="text-sm">UI Designer</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="card bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-8 mb-12" style={{ '--glow-x': '50%', '--glow-y': '50%', '--glow-intensity': 0, position: 'relative' }}>
            <div style={{ position: 'relative', zIndex: 1 }}>
              <h3 className="text-2xl font-bold mb-4">Contact</h3>
              <p className="text-gray-300 leading-relaxed">
                If you have any questions or feedback, please contact us on
                Instagram of one of the developers.
              </p>
              <div className="mt-8">
                <a
                  href="https://github.com/kullal/Premier-League-Prediction-Using-XGBoost"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 py-3 px-8 bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-lg font-semibold text-white rounded-full hover:bg-white/10 transition-colors duration-300 drop-shadow-lg"
                >
                  <svg
                    className="w-5 h-5"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      fillRule="evenodd"
                      d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                      clipRule="evenodd"
                    />
                  </svg>
                  View Source Code
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
