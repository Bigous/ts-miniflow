import { Sigmoid } from '../src/sigmoid';

test('Should greet with message', () => {
  const greeter = new Sigmoid('friend');
  expect(greeter.greet()).toBe('Bonjour, friend!');
});
