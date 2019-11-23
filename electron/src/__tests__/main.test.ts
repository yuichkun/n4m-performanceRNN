import App from '../main'

describe('App', () => {
  let app: App;

  beforeEach(() => {
    app = new App();
  })

  describe('constructor', () => {
    it('app is defined', () => {
      expect(App).toBeDefined()
      expect(app).toBeInstanceOf(App)
    })
  })

  describe('initialize', () => {
    it('loads models', async () => {
      expect(app.models).toBeUndefined();
      await app.initialize()
      expect(app.models).toBeDefined();
    })
    it('sets loop id at 0', async () => {
      app.currentLoopId = 100;
      expect(app.currentLoopId).toBe(100)
      await app.initialize()
      expect(app.currentLoopId).toBe(0)
    })
  })
})